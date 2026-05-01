/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * NIST-developed software is provided by NIST as a public service. You may
 * use, copy and distribute copies of the software in any medium, provided that
 * you keep intact this entire notice. You may improve, modify and create
 * derivative works of the software or any portion of the software, and you may
 * copy and distribute such modifications or works. Modified works should carry
 * a notice stating that you changed the software and should note the date and
 * nature of any such change. Please explicitly acknowledge the National
 * Institute of Standards and Technology as the source of the software.
 *
 * NIST-developed software is expressly provided "AS IS." NIST MAKES NO
 * WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF
 * LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST
 * NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
 * UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST
 * DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE
 * SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE
 * CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
 *
 * You are solely responsible for determining the appropriateness of using and
 * distributing the software and you assume all risks associated with its use,
 * including but not limited to the risks and costs of program errors,
 * compliance with applicable laws, damage to or loss of data, programs or
 * equipment, and the unavailability or interruption of operation. This
 * software is not intended to be used in any situation where a failure could
 * cause risk of injury or damage to property. The software developed by NIST
 * employees is not subject to copyright protection within the United States.
 */

#include <ns3/applications-module.h>
#include <ns3/core-module.h>
#include <ns3/internet-module.h>
#include <ns3/lte-module.h>
#include <ns3/mobility-module.h>
#include <ns3/oran-module.h>
#include <ns3/point-to-point-module.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <sqlite3.h>
#include <algorithm>
#include <unordered_set>

#include "ns3/lte-hex-grid-enb-topology-helper.h"
#include "ns3/oran-reporter-apploss.h"
#include "ns3/oran-helper.h"
#include "ns3/oran-near-rt-ric.h"
#include "ns3/oran-data-repository-sqlite.h"
#include "ns3/oran-cmm-noop.h"
#include "ns3/oran-lm.h"
#include "ns3/oran-reporter-location.h"
#include "ns3/oran-reporter-lte-ue-cell-info.h"
#include "ns3/oran-reporter-lte-ue-rsrp-rsrq.h"
#include "ns3/oran-e2-node-terminator-lte-ue.h"
#include "ns3/oran-e2-node-terminator-lte-enb.h"

#include "ns3/lte-ue-net-device.h"
#include "ns3/lte-enb-net-device.h"
#include "ns3/lte-ue-phy.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LteOranHexGridLstmHybrid");

/**
 * LTE + O-RAN hex-grid baseline scenario.
 *
 * 7 LTE sites with 3 sectors each (21 cells) serve 30 mobile UEs.
 * Built-in LTE A3 RSRP handover is used as the mobility baseline.
 * When O-RAN is enabled, the Near-RT RIC collects UE location,
 * serving-cell information, and application-loss statistics through
 * E2 node terminators and stores them in the SQLite data repository.
 */

#include <unordered_map>
#include <vector>
//count ping-pong


struct UeHoState
{
    uint16_t pendingSourceCellId = 0;
    uint16_t pendingTargetCellId = 0;

    uint16_t lastCompletedSourceCellId = 0;
    uint16_t lastCompletedTargetCellId = 0;
    Time lastCompletedHoTime = Seconds(0);

    uint32_t successfulHoCount = 0;
    uint32_t pingPongCount = 0;
};

static std::unordered_map<uint64_t, UeHoState> g_ueHoStates;
static Time g_pingPongWindow = Seconds(5);

enum class TrafficDirection
{
    DL_TX,
    DL_RX,
    UL_TX,
    UL_RX,
};

struct IntervalTrafficStats
{
    uint64_t dlTxBytes = 0;
    uint64_t dlRxBytes = 0;
    uint64_t ulTxBytes = 0;
    uint64_t ulRxBytes = 0;
    uint32_t dlTxPackets = 0;
    uint32_t dlRxPackets = 0;
    uint32_t ulTxPackets = 0;
    uint32_t ulRxPackets = 0;
};

struct ServingRadioState
{
    bool valid = false;
    uint16_t cellId = 0;
    uint16_t rnti = 0;
    double rsrp = std::numeric_limits<double>::quiet_NaN();
    double sinr = std::numeric_limits<double>::quiet_NaN();
    uint8_t ccId = 0;
    Time time = Seconds(0);
};

struct NeighborMeasurement
{
    uint16_t cellId = 0;
    double rsrp = -std::numeric_limits<double>::infinity();
    double rsrq = -std::numeric_limits<double>::infinity();
    bool isServingCell = false;
    uint8_t ccId = 0;
    Time time = Seconds(0);
};

struct RecentHoInfo
{
    Time lastStartTime = Seconds(-1);
    Time lastEndTime = Seconds(-1);
    uint16_t lastStartSourceCell = 0;
    uint16_t lastStartTargetCell = 0;
    uint16_t lastCompletedTargetCell = 0;
    bool lastPingPong = false;
    bool handoverInProgress = false;
};

struct PendingLstmRequest
{
    bool active = false;
    Time requestTime = Seconds(0);
    uint16_t sourceCellId = 0;
    uint16_t targetCellId = 0;
    double confidence = 0.0;
};

struct LstmDecision
{
    uint64_t imsi = 0;
    uint16_t servingCellId = 0;
    uint16_t targetCellId = 0;
    double confidence = 0.0;
    std::string status;
    std::string reason;
};

class HybridLstmController
{
  public:
    HybridLstmController(const std::string& dbPath,
                         const std::string& runDir,
                         Ptr<LteHelper> lteHelper,
                         NetDeviceContainer ueLteDevs,
                         NetDeviceContainer enbLteDevs,
                         bool enableLstmController,
                         double decisionIntervalSec,
                         uint32_t seqLen,
                         double minConfidence,
                         double cooldownSec,
                         double triggerThreshold,
                         double targetThreshold,
                         double utilityThreshold,
                         uint32_t targetDistanceTopK,
                         bool preferNonServingTarget,
                         const std::string& pythonPath,
                         const std::string& inferenceScript,
                         const std::string& checkpointPath)
        : m_dbPath(dbPath),
          m_runDir(runDir),
          m_lteHelper(lteHelper),
          m_ueLteDevs(ueLteDevs),
          m_enbLteDevs(enbLteDevs),
          m_enableLstmController(enableLstmController),
          m_decisionInterval(Seconds(decisionIntervalSec)),
          m_seqLen(seqLen),
          m_minConfidence(minConfidence),
          m_cooldown(Seconds(cooldownSec)),
          m_triggerThreshold(triggerThreshold),
          m_targetThreshold(targetThreshold),
          m_utilityThreshold(utilityThreshold),
          m_targetDistanceTopK(targetDistanceTopK),
          m_preferNonServingTarget(preferNonServingTarget),
          m_pythonPath(pythonPath),
          m_inferenceScript(inferenceScript),
          m_checkpointPath(checkpointPath),
          m_decisionCsvPath(runDir + "/lstm-decisions.csv"),
          m_db(nullptr)
    {
        for (NetDeviceContainer::Iterator it = m_ueLteDevs.Begin(); it != m_ueLteDevs.End(); ++it)
        {
            Ptr<LteUeNetDevice> ueDev = (*it)->GetObject<LteUeNetDevice>();
            if (ueDev)
            {
                m_imsiToUeDev[ueDev->GetImsi()] = ueDev;
            }
        }

        for (NetDeviceContainer::Iterator it = m_enbLteDevs.Begin(); it != m_enbLteDevs.End(); ++it)
        {
            Ptr<LteEnbNetDevice> enbDev = (*it)->GetObject<LteEnbNetDevice>();
            if (enbDev)
            {
                m_cellIdToEnbDev[enbDev->GetCellId()] = *it;
            }
        }
    }

    ~HybridLstmController()
    {
        if (m_db != nullptr)
        {
            sqlite3_close(m_db);
            m_db = nullptr;
        }
    }

    void Initialize()
    {
        NS_ABORT_MSG_IF(m_dbPath.empty(), "HybridLstmController requires a database path");
        OpenDb();
        InitializeTables();

        std::ofstream featuresTrace(m_runDir + "/lstm-features.tr", std::ios::out);
        featuresTrace
            << "time imsi ueId nodeId rnti servingCellId x y z dx dy speed servingRsrp "
               "servingSinr dlTxBytes dlRxBytes ulTxBytes ulRxBytes dlDeliveryRatio "
               "ulDeliveryRatio appLoss distServing bestNeighborCellId bestNeighborRsrp "
               "bestNeighborRsrq secondNeighborCellId secondNeighborRsrp secondNeighborRsrq "
               "hoStartFlag hoEndFlag pingPongFlag hoSourceCell hoTargetCell completedTargetCell"
            << std::endl;

        std::ofstream decisionsTrace(m_runDir + "/handover-decision-source.tr", std::ios::out);
        decisionsTrace
            << "time imsi source actualOrRequested servingCellId targetCellId confidence reason"
            << std::endl;

        std::ofstream controllerTrace(m_runDir + "/lstm-controller-state.tr", std::ios::out);
        controllerTrace
            << "time status reason totalDecisions readyDecisions"
            << std::endl;
    }

    void Start()
    {
        Simulator::Schedule(m_decisionInterval, &HybridLstmController::RunCycle, this);
    }

    void RecordTraffic(uint64_t imsi, TrafficDirection direction, uint32_t bytes)
    {
        auto& stats = m_intervalTraffic[imsi];
        switch (direction)
        {
        case TrafficDirection::DL_TX:
            stats.dlTxBytes += bytes;
            stats.dlTxPackets++;
            break;
        case TrafficDirection::DL_RX:
            stats.dlRxBytes += bytes;
            stats.dlRxPackets++;
            break;
        case TrafficDirection::UL_TX:
            stats.ulTxBytes += bytes;
            stats.ulTxPackets++;
            break;
        case TrafficDirection::UL_RX:
            stats.ulRxBytes += bytes;
            stats.ulRxPackets++;
            break;
        }
    }

    void UpdateServingRadio(uint64_t imsi,
                            uint16_t cellId,
                            uint16_t rnti,
                            double rsrp,
                            double sinr,
                            uint8_t ccId)
    {
        auto& state = m_servingRadio[imsi];
        state.valid = true;
        state.cellId = cellId;
        state.rnti = rnti;
        state.rsrp = rsrp;
        state.sinr = sinr;
        state.ccId = ccId;
        state.time = Simulator::Now();
    }

    void UpdateRsrpRsrq(uint64_t imsi,
                        uint16_t /*rnti*/,
                        uint16_t cellId,
                        double rsrp,
                        double rsrq,
                        bool isServingCell,
                        uint8_t ccId)
    {
        if (!std::isfinite(rsrp) || !std::isfinite(rsrq))
        {
            return;
        }

        auto& perCell = m_neighborMeasurements[imsi];
        auto& meas = perCell[cellId];
        meas.cellId = cellId;
        meas.rsrp = rsrp;
        meas.rsrq = rsrq;
        meas.isServingCell = isServingCell;
        meas.ccId = ccId;
        meas.time = Simulator::Now();
    }

    void OnHandoverStart(uint64_t imsi, uint16_t sourceCellId, uint16_t targetCellId)
    {
        auto& info = m_recentHoInfo[imsi];
        info.lastStartTime = Simulator::Now();
        info.lastStartSourceCell = sourceCellId;
        info.lastStartTargetCell = targetCellId;
        info.handoverInProgress = true;

        std::string actualSource = "A3";
        double confidence = 0.0;
        std::string reason = "fallback_or_lte";

        auto it = m_pendingRequests.find(imsi);
        if (it != m_pendingRequests.end() && it->second.active &&
            it->second.sourceCellId == sourceCellId && it->second.targetCellId == targetCellId &&
            (Simulator::Now() - it->second.requestTime) <= Seconds(1.5))
        {
            actualSource = "LSTM";
            confidence = it->second.confidence;
            reason = "matched_pending_request";
            it->second.active = false;
        }

        AppendDecisionTrace(actualSource,
                            "HANDOVER_START",
                            imsi,
                            sourceCellId,
                            targetCellId,
                            confidence,
                            reason);
    }

    void OnHandoverEnd(uint64_t imsi, uint16_t targetCellId, bool isPingPong)
    {
        auto& info = m_recentHoInfo[imsi];
        info.lastEndTime = Simulator::Now();
        info.lastCompletedTargetCell = targetCellId;
        info.lastPingPong = isPingPong;
        info.handoverInProgress = false;
    }

  private:
    void RunCycle()
    {
        CleanupStaleRequests();
        SampleAndPersistFeatures();

        std::vector<LstmDecision> decisions;
        uint32_t readyDecisions = 0;
        std::string cycleStatus = "A3_FALLBACK";
        std::string cycleReason = "controller_disabled";

        if (m_enableLstmController && InvokeInference(cycleReason))
        {
            decisions = ReadDecisionCsv();
            readyDecisions = std::count_if(decisions.begin(),
                                           decisions.end(),
                                           [](const LstmDecision& decision) {
                                               return decision.status == "ready";
                                           });

            if (!decisions.empty() && readyDecisions > 0)
            {
                cycleStatus = "LSTM_ACTIVE";
                cycleReason = "ready_decisions_present";
            }
            else if (!decisions.empty())
            {
                cycleStatus = "A3_FALLBACK";
                cycleReason = "no_ready_decisions";
            }
            else
            {
                cycleStatus = "A3_FALLBACK";
                cycleReason = "empty_decision_file";
            }

            ApplyDecisions(decisions);
        }

        AppendControllerState(cycleStatus, cycleReason, decisions.size(), readyDecisions);

        Simulator::Schedule(m_decisionInterval, &HybridLstmController::RunCycle, this);
    }

    void CleanupStaleRequests()
    {
        for (auto& [imsi, pending] : m_pendingRequests)
        {
            if (pending.active && (Simulator::Now() - pending.requestTime) > Seconds(2.0))
            {
                pending.active = false;
            }
        }
    }

    static std::string ShellQuote(const std::string& value)
    {
        std::string quoted = "'";
        for (char c : value)
        {
            if (c == '\'')
            {
                quoted += "'\\''";
            }
            else
            {
                quoted += c;
            }
        }
        quoted += "'";
        return quoted;
    }

    void OpenDb()
    {
        if (m_db != nullptr)
        {
            return;
        }

        int rc = sqlite3_open(m_dbPath.c_str(), &m_db);
        NS_ABORT_MSG_IF(rc != SQLITE_OK, "Could not open SQLite DB for hybrid controller: " + m_dbPath);
        sqlite3_busy_timeout(m_db, 5000);
    }

    void ExecSql(const std::string& sql)
    {
        char* err = nullptr;
        int rc = sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, &err);
        if (rc != SQLITE_OK)
        {
            std::string errMsg = err ? err : "unknown sqlite error";
            sqlite3_free(err);
            NS_ABORT_MSG("SQLite exec failed: " + errMsg + " sql=" + sql);
        }
    }

    void InitializeTables()
    {
        ExecSql("CREATE TABLE IF NOT EXISTS lstm_features ("
                "simulationtime REAL,"
                "imsi INTEGER,"
                "ueid INTEGER,"
                "nodeid INTEGER,"
                "rnti INTEGER,"
                "servingcellid INTEGER,"
                "x REAL,y REAL,z REAL,"
                "dx REAL,dy REAL,speed REAL,"
                "servingrsrp REAL,servingsinr REAL,"
                "dltxbytes INTEGER,dlrxbytes INTEGER,ultxbytes INTEGER,ulrxbytes INTEGER,"
                "dldeliveryratio REAL,uldeliveryratio REAL,"
                "apploss REAL,distserving REAL,"
                "bestneighborcellid INTEGER,bestneighborrsrp REAL,bestneighborrsrq REAL,"
                "secondneighborcellid INTEGER,secondneighborrsrp REAL,secondneighborrsrq REAL,"
                "hostartflag INTEGER,hoendflag INTEGER,pingpongflag INTEGER,"
                "hosourcecell INTEGER,hotargetcell INTEGER,completedtargetcell INTEGER"
                ");");

        ExecSql("CREATE TABLE IF NOT EXISTS handover_decisions ("
                "simulationtime REAL,"
                "imsi INTEGER,"
                "source TEXT,"
                "action TEXT,"
                "servingcellid INTEGER,"
                "targetcellid INTEGER,"
                "confidence REAL,"
                "reason TEXT"
                ");");

        ExecSql("CREATE TABLE IF NOT EXISTS lstm_controller_state ("
                "simulationtime REAL,"
                "status TEXT,"
                "reason TEXT,"
                "totaldecisions INTEGER,"
                "readydecisions INTEGER"
                ");");
    }

    void AppendDecisionTrace(const std::string& source,
                             const std::string& action,
                             uint64_t imsi,
                             uint16_t servingCellId,
                             uint16_t targetCellId,
                             double confidence,
                             const std::string& reason)
    {
        {
            std::ofstream out(m_runDir + "/handover-decision-source.tr", std::ios::app);
            out << Simulator::Now().GetSeconds() << " "
                << imsi << " "
                << source << " "
                << action << " "
                << servingCellId << " "
                << targetCellId << " "
                << confidence << " "
                << reason
                << std::endl;
        }

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(
            m_db,
            "INSERT INTO handover_decisions "
            "(simulationtime, imsi, source, action, servingcellid, targetcellid, confidence, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            -1,
            &stmt,
            0);

        sqlite3_bind_double(stmt, 1, Simulator::Now().GetSeconds());
        sqlite3_bind_int64(stmt, 2, imsi);
        sqlite3_bind_text(stmt, 3, source.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, action.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 5, servingCellId);
        sqlite3_bind_int(stmt, 6, targetCellId);
        sqlite3_bind_double(stmt, 7, confidence);
        sqlite3_bind_text(stmt, 8, reason.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    void AppendControllerState(const std::string& status,
                               const std::string& reason,
                               std::size_t totalDecisions,
                               std::size_t readyDecisions)
    {
        {
            std::ofstream out(m_runDir + "/lstm-controller-state.tr", std::ios::app);
            out << Simulator::Now().GetSeconds() << " "
                << status << " "
                << reason << " "
                << totalDecisions << " "
                << readyDecisions
                << std::endl;
        }

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(m_db,
                           "INSERT INTO lstm_controller_state "
                           "(simulationtime, status, reason, totaldecisions, readydecisions) "
                           "VALUES (?, ?, ?, ?, ?);",
                           -1,
                           &stmt,
                           0);

        sqlite3_bind_double(stmt, 1, Simulator::Now().GetSeconds());
        sqlite3_bind_text(stmt, 2, status.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, reason.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 4, static_cast<sqlite3_int64>(totalDecisions));
        sqlite3_bind_int64(stmt, 5, static_cast<sqlite3_int64>(readyDecisions));
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    void SampleAndPersistFeatures()
    {
        const double nowSec = Simulator::Now().GetSeconds();

        for (uint32_t idx = 0; idx < m_ueLteDevs.GetN(); ++idx)
        {
            Ptr<LteUeNetDevice> ueDev = m_ueLteDevs.Get(idx)->GetObject<LteUeNetDevice>();
            if (!ueDev)
            {
                continue;
            }

            Ptr<Node> ueNode = ueDev->GetNode();
            Ptr<MobilityModel> mob = ueNode->GetObject<MobilityModel>();
            if (!mob)
            {
                continue;
            }

            const uint64_t imsi = ueDev->GetImsi();
            Ptr<LteUeRrc> rrc = ueDev->GetRrc();
            const uint16_t servingCellId = rrc ? rrc->GetCellId() : 0;
            const uint16_t rnti = rrc ? rrc->GetRnti() : 0;

            Vector pos = mob->GetPosition();
            Vector prev = m_lastPositions.count(imsi) ? m_lastPositions[imsi] : pos;
            Time prevTime = m_lastPositionTime.count(imsi) ? m_lastPositionTime[imsi] : Simulator::Now();
            double dt = std::max(1e-9, (Simulator::Now() - prevTime).GetSeconds());
            double dx = pos.x - prev.x;
            double dy = pos.y - prev.y;
            double speed = std::sqrt(dx * dx + dy * dy) / dt;

            m_lastPositions[imsi] = pos;
            m_lastPositionTime[imsi] = Simulator::Now();

            IntervalTrafficStats traffic = m_intervalTraffic[imsi];
            double dlDeliveryRatio =
                traffic.dlTxBytes > 0 ? static_cast<double>(traffic.dlRxBytes) / traffic.dlTxBytes : 0.0;
            double ulDeliveryRatio =
                traffic.ulTxBytes > 0 ? static_cast<double>(traffic.ulRxBytes) / traffic.ulTxBytes : 0.0;
            double appLoss =
                traffic.dlTxPackets > 0
                    ? static_cast<double>(traffic.dlTxPackets - std::min(traffic.dlTxPackets, traffic.dlRxPackets)) /
                          static_cast<double>(traffic.dlTxPackets)
                    : 0.0;

            auto radioIt = m_servingRadio.find(imsi);
            double servingRsrp = std::numeric_limits<double>::quiet_NaN();
            double servingSinr = std::numeric_limits<double>::quiet_NaN();
            if (radioIt != m_servingRadio.end() && radioIt->second.valid)
            {
                servingRsrp = radioIt->second.rsrp;
                servingSinr = radioIt->second.sinr;
            }

            double distServing = std::numeric_limits<double>::quiet_NaN();
            auto enbIt = m_cellIdToEnbDev.find(servingCellId);
            if (enbIt != m_cellIdToEnbDev.end())
            {
                Ptr<LteEnbNetDevice> enbDev = enbIt->second->GetObject<LteEnbNetDevice>();
                Ptr<MobilityModel> enbMob = enbDev ? enbDev->GetNode()->GetObject<MobilityModel>() : nullptr;
                if (enbMob)
                {
                    Vector enbPos = enbMob->GetPosition();
                    distServing = std::sqrt(std::pow(pos.x - enbPos.x, 2) + std::pow(pos.y - enbPos.y, 2));
                }
            }

            uint16_t bestNeighborCellId = 0;
            double bestNeighborRsrp = -std::numeric_limits<double>::infinity();
            double bestNeighborRsrq = -std::numeric_limits<double>::infinity();
            uint16_t secondNeighborCellId = 0;
            double secondNeighborRsrp = -std::numeric_limits<double>::infinity();
            double secondNeighborRsrq = -std::numeric_limits<double>::infinity();

            auto neighIt = m_neighborMeasurements.find(imsi);
            if (neighIt != m_neighborMeasurements.end())
            {
                std::vector<NeighborMeasurement> ordered;
                ordered.reserve(neighIt->second.size());
                for (const auto& kv : neighIt->second)
                {
                    if (!kv.second.isServingCell)
                    {
                        ordered.push_back(kv.second);
                    }
                }
                std::sort(ordered.begin(),
                          ordered.end(),
                          [](const NeighborMeasurement& a, const NeighborMeasurement& b) {
                              return a.rsrp > b.rsrp;
                          });
                if (!ordered.empty())
                {
                    bestNeighborCellId = ordered[0].cellId;
                    bestNeighborRsrp = ordered[0].rsrp;
                    bestNeighborRsrq = ordered[0].rsrq;
                }
                if (ordered.size() > 1)
                {
                    secondNeighborCellId = ordered[1].cellId;
                    secondNeighborRsrp = ordered[1].rsrp;
                    secondNeighborRsrq = ordered[1].rsrq;
                }
            }

            auto hoIt = m_recentHoInfo.find(imsi);
            int hoStartFlag = 0;
            int hoEndFlag = 0;
            int pingPongFlag = 0;
            uint16_t hoSourceCell = 0;
            uint16_t hoTargetCell = 0;
            uint16_t completedTargetCell = 0;
            if (hoIt != m_recentHoInfo.end())
            {
                hoStartFlag = (Simulator::Now() - hoIt->second.lastStartTime) <= m_decisionInterval ? 1 : 0;
                hoEndFlag = (Simulator::Now() - hoIt->second.lastEndTime) <= m_decisionInterval ? 1 : 0;
                pingPongFlag = hoIt->second.lastPingPong && hoEndFlag ? 1 : 0;
                hoSourceCell = hoIt->second.lastStartSourceCell;
                hoTargetCell = hoIt->second.lastStartTargetCell;
                completedTargetCell = hoIt->second.lastCompletedTargetCell;
            }

            sqlite3_stmt* stmt = nullptr;
            sqlite3_prepare_v2(
                m_db,
                "INSERT INTO lstm_features "
                "(simulationtime, imsi, ueid, nodeid, rnti, servingcellid, x, y, z, dx, dy, speed, "
                "servingrsrp, servingsinr, dltxbytes, dlrxbytes, ultxbytes, ulrxbytes, "
                "dldeliveryratio, uldeliveryratio, apploss, distserving, "
                "bestneighborcellid, bestneighborrsrp, bestneighborrsrq, "
                "secondneighborcellid, secondneighborrsrp, secondneighborrsrq, "
                "hostartflag, hoendflag, pingpongflag, hosourcecell, hotargetcell, completedtargetcell) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                -1,
                &stmt,
                0);

            sqlite3_bind_double(stmt, 1, nowSec);
            sqlite3_bind_int64(stmt, 2, imsi);
            sqlite3_bind_int(stmt, 3, idx);
            sqlite3_bind_int(stmt, 4, ueNode->GetId());
            sqlite3_bind_int(stmt, 5, rnti);
            sqlite3_bind_int(stmt, 6, servingCellId);
            sqlite3_bind_double(stmt, 7, pos.x);
            sqlite3_bind_double(stmt, 8, pos.y);
            sqlite3_bind_double(stmt, 9, pos.z);
            sqlite3_bind_double(stmt, 10, dx);
            sqlite3_bind_double(stmt, 11, dy);
            sqlite3_bind_double(stmt, 12, speed);
            sqlite3_bind_double(stmt, 13, servingRsrp);
            sqlite3_bind_double(stmt, 14, servingSinr);
            sqlite3_bind_int64(stmt, 15, traffic.dlTxBytes);
            sqlite3_bind_int64(stmt, 16, traffic.dlRxBytes);
            sqlite3_bind_int64(stmt, 17, traffic.ulTxBytes);
            sqlite3_bind_int64(stmt, 18, traffic.ulRxBytes);
            sqlite3_bind_double(stmt, 19, dlDeliveryRatio);
            sqlite3_bind_double(stmt, 20, ulDeliveryRatio);
            sqlite3_bind_double(stmt, 21, appLoss);
            sqlite3_bind_double(stmt, 22, distServing);
            sqlite3_bind_int(stmt, 23, bestNeighborCellId);
            sqlite3_bind_double(stmt, 24, bestNeighborRsrp);
            sqlite3_bind_double(stmt, 25, bestNeighborRsrq);
            sqlite3_bind_int(stmt, 26, secondNeighborCellId);
            sqlite3_bind_double(stmt, 27, secondNeighborRsrp);
            sqlite3_bind_double(stmt, 28, secondNeighborRsrq);
            sqlite3_bind_int(stmt, 29, hoStartFlag);
            sqlite3_bind_int(stmt, 30, hoEndFlag);
            sqlite3_bind_int(stmt, 31, pingPongFlag);
            sqlite3_bind_int(stmt, 32, hoSourceCell);
            sqlite3_bind_int(stmt, 33, hoTargetCell);
            sqlite3_bind_int(stmt, 34, completedTargetCell);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);

            {
                std::ofstream trace(m_runDir + "/lstm-features.tr", std::ios::app);
                trace << nowSec << " "
                      << imsi << " "
                      << idx << " "
                      << ueNode->GetId() << " "
                      << rnti << " "
                      << servingCellId << " "
                      << pos.x << " "
                      << pos.y << " "
                      << pos.z << " "
                      << dx << " "
                      << dy << " "
                      << speed << " "
                      << servingRsrp << " "
                      << servingSinr << " "
                      << traffic.dlTxBytes << " "
                      << traffic.dlRxBytes << " "
                      << traffic.ulTxBytes << " "
                      << traffic.ulRxBytes << " "
                      << dlDeliveryRatio << " "
                      << ulDeliveryRatio << " "
                      << appLoss << " "
                      << distServing << " "
                      << bestNeighborCellId << " "
                      << bestNeighborRsrp << " "
                      << bestNeighborRsrq << " "
                      << secondNeighborCellId << " "
                      << secondNeighborRsrp << " "
                      << secondNeighborRsrq << " "
                      << hoStartFlag << " "
                      << hoEndFlag << " "
                      << pingPongFlag << " "
                      << hoSourceCell << " "
                      << hoTargetCell << " "
                      << completedTargetCell
                      << std::endl;
            }
        }

        m_intervalTraffic.clear();
    }

    bool InvokeInference(std::string& reason)
    {
        if (!std::filesystem::exists(m_pythonPath))
        {
            reason = "missing_python";
            return false;
        }

        if (!std::filesystem::exists(m_inferenceScript))
        {
            reason = "missing_inference_script";
            return false;
        }

        if (m_checkpointPath.empty() || !std::filesystem::exists(m_checkpointPath))
        {
            reason = "missing_checkpoint";
            return false;
        }

        std::string cmd = ShellQuote(m_pythonPath) + " " + ShellQuote(m_inferenceScript) +
                          " --db-path " + ShellQuote(m_dbPath) +
                          " --checkpoint-path " + ShellQuote(m_checkpointPath) +
                          " --output-path " + ShellQuote(m_decisionCsvPath) +
                          " --seq-len " + std::to_string(m_seqLen);

        if (m_triggerThreshold >= 0.0)
        {
            cmd += " --trigger-threshold " + std::to_string(m_triggerThreshold);
        }

        if (m_targetThreshold >= 0.0)
        {
            cmd += " --target-threshold " + std::to_string(m_targetThreshold);
        }

        if (m_utilityThreshold >= 0.0)
        {
            cmd += " --utility-threshold " + std::to_string(m_utilityThreshold);
        }

        if (m_preferNonServingTarget)
        {
            cmd += " --prefer-non-serving-target";
        }

        int rc = std::system(cmd.c_str());
        if (rc != 0)
        {
            reason = "inference_failed";
            return false;
        }

        if (!std::filesystem::exists(m_decisionCsvPath))
        {
            reason = "missing_decision_csv";
            return false;
        }

        reason = "inference_ok";
        return true;
    }

    std::vector<LstmDecision> ReadDecisionCsv() const
    {
        std::vector<LstmDecision> decisions;
        std::ifstream in(m_decisionCsvPath);
        if (!in.is_open())
        {
            return decisions;
        }

        std::string line;
        std::getline(in, line); // header
        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }

            std::stringstream ss(line);
            std::string token;
            LstmDecision d;

            std::getline(ss, token, ',');
            d.imsi = static_cast<uint64_t>(std::stoull(token));
            std::getline(ss, token, ',');
            d.servingCellId = static_cast<uint16_t>(std::stoul(token));
            std::getline(ss, token, ',');
            d.targetCellId = static_cast<uint16_t>(std::stoul(token));
            std::getline(ss, token, ',');
            d.confidence = std::stod(token);
            std::getline(ss, d.status, ',');
            std::getline(ss, d.reason);

            decisions.push_back(d);
        }
        return decisions;
    }

    void ApplyDecisions(const std::vector<LstmDecision>& decisions)
    {
        for (const auto& decision : decisions)
        {
            auto ueIt = m_imsiToUeDev.find(decision.imsi);
            if (ueIt == m_imsiToUeDev.end())
            {
                continue;
            }

            Ptr<LteUeNetDevice> ueDev = ueIt->second;
            Ptr<LteUeRrc> ueRrc = ueDev->GetRrc();
            if (!ueRrc)
            {
                continue;
            }

            const uint16_t currentCellId = ueRrc->GetCellId();
            if (currentCellId == 0)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    0,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "missing_serving_cell");
                continue;
            }

            if (decision.status != "ready")
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    decision.reason.empty() ? decision.status : decision.reason);
                continue;
            }

            if (decision.targetCellId == 0)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    0,
                                    decision.confidence,
                                    "invalid_target_cell");
                continue;
            }

            if (decision.targetCellId == currentCellId)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "same_as_serving_cell");
                continue;
            }

            if (decision.confidence < m_minConfidence)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "low_confidence");
                continue;
            }

            auto pendingIt = m_pendingRequests.find(decision.imsi);
            if (pendingIt != m_pendingRequests.end() && pendingIt->second.active)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "pending_request_active");
                continue;
            }

            auto hoIt = m_recentHoInfo.find(decision.imsi);
            if (hoIt != m_recentHoInfo.end() && hoIt->second.handoverInProgress)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "handover_in_progress");
                continue;
            }

            if (hoIt != m_recentHoInfo.end() &&
                hoIt->second.lastEndTime >= Seconds(0) &&
                (Simulator::Now() - hoIt->second.lastEndTime) < m_cooldown)
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "cooldown");
                continue;
            }

            auto sourceEnbIt = m_cellIdToEnbDev.find(currentCellId);
            auto targetEnbIt = m_cellIdToEnbDev.find(decision.targetCellId);
            if (sourceEnbIt == m_cellIdToEnbDev.end() || targetEnbIt == m_cellIdToEnbDev.end())
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "unknown_enb_mapping");
                continue;
            }

            Ptr<MobilityModel> ueMob = ueDev->GetNode()->GetObject<MobilityModel>();
            if (m_targetDistanceTopK > 0 && ueMob &&
                !IsTargetCellAllowed(ueMob->GetPosition(), decision.targetCellId))
            {
                AppendDecisionTrace("LSTM",
                                    "SKIP",
                                    decision.imsi,
                                    currentCellId,
                                    decision.targetCellId,
                                    decision.confidence,
                                    "target_outside_topk_distance");
                continue;
            }

            m_lteHelper->HandoverRequest(Seconds(0.0), ueDev, sourceEnbIt->second, decision.targetCellId);
            auto& pending = m_pendingRequests[decision.imsi];
            pending.active = true;
            pending.requestTime = Simulator::Now();
            pending.sourceCellId = currentCellId;
            pending.targetCellId = decision.targetCellId;
            pending.confidence = decision.confidence;

            AppendDecisionTrace("LSTM",
                                "REQUEST",
                                decision.imsi,
                                currentCellId,
                                decision.targetCellId,
                                decision.confidence,
                                decision.reason.empty() ? "python_model" : decision.reason);
        }
    }

    bool IsTargetCellAllowed(const Vector& uePosition, uint16_t targetCellId) const
    {
        if (m_targetDistanceTopK == 0)
        {
            return true;
        }

        std::vector<std::pair<double, uint16_t>> distances;
        distances.reserve(m_cellIdToEnbDev.size());

        for (const auto& [cellId, dev] : m_cellIdToEnbDev)
        {
            Ptr<LteEnbNetDevice> enbDev = dev->GetObject<LteEnbNetDevice>();
            Ptr<MobilityModel> enbMob = enbDev ? enbDev->GetNode()->GetObject<MobilityModel>() : nullptr;
            if (!enbMob)
            {
                continue;
            }

            Vector enbPos = enbMob->GetPosition();
            double d = std::sqrt(std::pow(uePosition.x - enbPos.x, 2) + std::pow(uePosition.y - enbPos.y, 2));
            distances.emplace_back(d, cellId);
        }

        std::sort(distances.begin(),
                  distances.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        const std::size_t topK = std::min<std::size_t>(m_targetDistanceTopK, distances.size());
        for (std::size_t i = 0; i < topK; ++i)
        {
            if (distances[i].second == targetCellId)
            {
                return true;
            }
        }

        return false;
    }

  private:
    std::string m_dbPath;
    std::string m_runDir;
    Ptr<LteHelper> m_lteHelper;
    NetDeviceContainer m_ueLteDevs;
    NetDeviceContainer m_enbLteDevs;
    bool m_enableLstmController;
    Time m_decisionInterval;
    uint32_t m_seqLen;
    double m_minConfidence;
    Time m_cooldown;
    double m_triggerThreshold;
    double m_targetThreshold;
    double m_utilityThreshold;
    uint32_t m_targetDistanceTopK;
    bool m_preferNonServingTarget;
    std::string m_pythonPath;
    std::string m_inferenceScript;
    std::string m_checkpointPath;
    std::string m_decisionCsvPath;
    sqlite3* m_db;

    std::unordered_map<uint64_t, Ptr<LteUeNetDevice>> m_imsiToUeDev;
    std::unordered_map<uint16_t, Ptr<NetDevice>> m_cellIdToEnbDev;
    std::unordered_map<uint64_t, IntervalTrafficStats> m_intervalTraffic;
    std::unordered_map<uint64_t, ServingRadioState> m_servingRadio;
    std::unordered_map<uint64_t, std::unordered_map<uint16_t, NeighborMeasurement>>
        m_neighborMeasurements;
    std::unordered_map<uint64_t, Vector> m_lastPositions;
    std::unordered_map<uint64_t, Time> m_lastPositionTime;
    std::unordered_map<uint64_t, RecentHoInfo> m_recentHoInfo;
    std::unordered_map<uint64_t, PendingLstmRequest> m_pendingRequests;
};

static std::unique_ptr<HybridLstmController> g_hybridLstmController;

static void
NotifyHybridHandoverStart(uint64_t imsi, uint16_t sourceCellId, uint16_t targetCellId)
{
    if (g_hybridLstmController)
    {
        g_hybridLstmController->OnHandoverStart(imsi, sourceCellId, targetCellId);
    }
}

static void
NotifyHybridHandoverEnd(uint64_t imsi, uint16_t targetCellId, bool isPingPong)
{
    if (g_hybridLstmController)
    {
        g_hybridLstmController->OnHandoverEnd(imsi, targetCellId, isPingPong);
    }
}

void
HandoverEndTraceDetailed(Ptr<OutputStreamWrapper> eventStream,
                         Ptr<OutputStreamWrapper> statsStream,
                         uint64_t imsi,
                         uint16_t cellid,
                         uint16_t rnti)
{
    auto& st = g_ueHoStates[imsi];

    st.successfulHoCount++;

    bool isPingPong = false;

    if (st.lastCompletedSourceCellId != 0 &&
        st.lastCompletedTargetCellId != 0 &&
        st.pendingSourceCellId == st.lastCompletedTargetCellId &&
        st.pendingTargetCellId == st.lastCompletedSourceCellId &&
        (Simulator::Now() - st.lastCompletedHoTime) <= g_pingPongWindow)
    {
        isPingPong = true;
        st.pingPongCount++;
    }

    *eventStream->GetStream()
        << Simulator::Now().GetSeconds() << " "
        << imsi << " "
        << cellid << " "
        << rnti << " "
        << st.successfulHoCount << " "
        << st.pingPongCount << " "
        << (isPingPong ? 1 : 0)
        << std::endl;

    *statsStream->GetStream()
        << Simulator::Now().GetSeconds() << " "
        << imsi << " "
        << st.successfulHoCount << " "
        << st.pingPongCount << " "
        << (st.successfulHoCount > 0
                ? static_cast<double>(st.pingPongCount) / st.successfulHoCount
                : 0.0)
        << std::endl;

    NotifyHybridHandoverEnd(imsi, cellid, isPingPong);

    st.lastCompletedSourceCellId = st.pendingSourceCellId;
    st.lastCompletedTargetCellId = st.pendingTargetCellId;
    st.lastCompletedHoTime = Simulator::Now();

    st.pendingSourceCellId = 0;
    st.pendingTargetCellId = 0;
}


void
UeCellStateTrace(Ptr<OutputStreamWrapper> stream,
                 NetDeviceContainer ueLteDevs,
                 NodeContainer ueNodes)
{
    for (uint32_t i = 0; i < ueLteDevs.GetN(); ++i)
    {
        Ptr<LteUeNetDevice> ueDev = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        if (!ueDev)
        {
            continue;
        }

        uint64_t imsi = ueDev->GetImsi();

        uint16_t rnti = 0;
        uint16_t servingCellId = 0;

        Ptr<LteUeRrc> ueRrc = ueDev->GetRrc();
        if (ueRrc)
        {
            rnti = ueRrc->GetRnti();
            servingCellId = ueRrc->GetCellId();
        }

        *stream->GetStream()
            << Simulator::Now().GetSeconds() << " "
            << i << " "
            << ueNodes.Get(i)->GetId() << " "
            << imsi << " "
            << servingCellId << " "
            << rnti
            << std::endl;
    }

    Simulator::Schedule(Seconds(1.0), &UeCellStateTrace, stream, ueLteDevs, ueNodes);
}


void
ConfigureDefaultLteTraceFiles(const std::string& runDir)
{
    Config::SetDefault("ns3::MacStatsCalculator::DlOutputFilename",
                       StringValue(runDir + "/DlMacStats.txt"));
    Config::SetDefault("ns3::MacStatsCalculator::UlOutputFilename",
                       StringValue(runDir + "/UlMacStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename",
                       StringValue(runDir + "/DlRlcStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename",
                       StringValue(runDir + "/UlRlcStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::DlPdcpOutputFilename",
                       StringValue(runDir + "/DlPdcpStats.txt"));
    Config::SetDefault("ns3::RadioBearerStatsCalculator::UlPdcpOutputFilename",
                       StringValue(runDir + "/UlPdcpStats.txt"));
    Config::SetDefault("ns3::PhyStatsCalculator::DlRsrpSinrFilename",
                       StringValue(runDir + "/DlRsrpSinrStats.txt"));
    Config::SetDefault("ns3::PhyStatsCalculator::UlSinrFilename",
                       StringValue(runDir + "/UlSinrStats.txt"));
    Config::SetDefault("ns3::PhyStatsCalculator::UlInterferenceFilename",
                       StringValue(runDir + "/UlInterferenceStats.txt"));
    Config::SetDefault("ns3::PhyTxStatsCalculator::DlTxOutputFilename",
                       StringValue(runDir + "/DlTxPhyStats.txt"));
    Config::SetDefault("ns3::PhyTxStatsCalculator::UlTxOutputFilename",
                       StringValue(runDir + "/UlTxPhyStats.txt"));
    Config::SetDefault("ns3::PhyRxStatsCalculator::DlRxOutputFilename",
                       StringValue(runDir + "/DlRxPhyStats.txt"));
    Config::SetDefault("ns3::PhyRxStatsCalculator::UlRxOutputFilename",
                       StringValue(runDir + "/UlRxPhyStats.txt"));
}


void
HandoverStartTrace(Ptr<OutputStreamWrapper> stream,
                   uint64_t imsi,
                   uint16_t sourceCellId,
                   uint16_t rnti,
                   uint16_t targetCellId)
{
    auto& st = g_ueHoStates[imsi];
    st.pendingSourceCellId = sourceCellId;
    st.pendingTargetCellId = targetCellId;

    *stream->GetStream()
        << Simulator::Now().GetSeconds() << " "
        << imsi << " "
        << sourceCellId << " "
        << targetCellId << " "
        << rnti
        << std::endl;

    NotifyHybridHandoverStart(imsi, sourceCellId, targetCellId);
}

void
SaveEnbPositionsToFile(NodeContainer enbNodes,
                       NetDeviceContainer enbLteDevs,
                       const std::string& filename)
{
    std::ofstream out(filename, std::ios::out);

    if (!out.is_open())
    {
        std::cerr << "Cannot open " << filename << std::endl;
        return;
    }

    out << "time,nodeId,cellId,x,y,z" << std::endl;

    double now = Simulator::Now().GetSeconds();

    for (uint32_t i = 0; i < enbNodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mob = enbNodes.Get(i)->GetObject<MobilityModel>();
        Ptr<LteEnbNetDevice> enbDev = nullptr;

        if (i < enbLteDevs.GetN())
        {
            enbDev = enbLteDevs.Get(i)->GetObject<LteEnbNetDevice>();
        }

        if (mob)
        {
            Vector p = mob->GetPosition();
            out << now << "," << enbNodes.Get(i)->GetId() << ",";
            if (enbDev)
            {
                out << enbDev->GetCellId();
            }
            else
            {
                out << "NO_LTE_ENB";
            }
            out << "," << p.x << "," << p.y << "," << p.z << std::endl;
        }
        else
        {
            out << now << "," << enbNodes.Get(i)->GetId() << ",";
            if (enbDev)
            {
                out << enbDev->GetCellId();
            }
            else
            {
                out << "NO_LTE_ENB";
            }
            out << ",NO_MOBILITY,NO_MOBILITY,NO_MOBILITY" << std::endl;
        }
    }

    out.close();
}

// Tracing current-cell RSRP and SINR
static void
TraceCurrentCellRsrpSinr(Ptr<OutputStreamWrapper> stream,
                         uint64_t imsi,
                         uint16_t cellId,
                         uint16_t rnti,
                         double rsrp,
                         double sinr,
                         uint8_t ccId)
{
    *stream->GetStream()
        << Simulator::Now().GetSeconds() << " "
        << rnti << " "
        << cellId << " "
        << rsrp << " "
        << sinr << " "
        << +ccId
        << std::endl;

    if (g_hybridLstmController)
    {
        g_hybridLstmController->UpdateServingRadio(imsi, cellId, rnti, rsrp, sinr, ccId);
    }
}

static void
TraceUeMeasurementsRsrpRsrq(uint64_t imsi,
                            uint16_t rnti,
                            uint16_t cellId,
                            double rsrp,
                            double rsrq,
                            bool isServingCell,
                            uint8_t ccId)
{
    if (g_hybridLstmController)
    {
        g_hybridLstmController->UpdateRsrpRsrq(
            imsi, rnti, cellId, rsrp, rsrq, isServingCell, ccId);
    }
}


// Trace each node's location
void
PositionTrace(Ptr<OutputStreamWrapper> stream, NodeContainer nodes)
{
    for (uint32_t i = 0; i < nodes.GetN(); i++)
    {
        Ptr<MobilityModel> mob = nodes.Get(i)->GetObject<MobilityModel>();
        if (!mob)
        {
            continue;
        }

        Vector pos = mob->GetPosition();

        *stream->GetStream()
            << Simulator::Now().GetSeconds() << " "
            << i << " "
            << nodes.Get(i)->GetId() << " "
            << pos.x << " "
            << pos.y << " "
            << pos.z
            << std::endl;
    }

    Simulator::Schedule(Seconds(1), &PositionTrace, stream, nodes);
}

// Trace handover events


// Output DB queries
void
QueryRcSink(std::string query, std::string args, int rc)
{
    std::cout << Simulator::Now().GetSeconds() << " Query "
              << ((rc == SQLITE_OK || rc == SQLITE_DONE) ? "OK" : "ERROR") << "(" << rc << "): \""
              << query << "\"";

    if (!args.empty())
    {
        std::cout << " (" << args << ")";
    }

    std::cout << std::endl;
}




namespace fs = std::filesystem;

static std::string
ResolvePath(const std::string& path)
{
    if (path.empty())
    {
        return path;
    }

    fs::path fsPath(path);
    if (fsPath.is_absolute())
    {
        return fsPath.lexically_normal().string();
    }

    return fs::absolute(fsPath).lexically_normal().string();
}

static std::string
FindNextRunDir(const std::string& baseDir, const std::string& prefix = "run")
{
    fs::create_directories(baseDir);

    for (uint32_t i = 1; i < 1000000; ++i)
    {
        std::ostringstream oss;
        oss << baseDir << "/" << prefix << "-" << std::setw(5) << std::setfill('0') << i;
        if (!fs::exists(oss.str()))
        {
            fs::create_directories(oss.str());
            return oss.str();
        }
    }

    NS_ABORT_MSG("Could not allocate a new run directory");
    return "";
}

static void
WriteRunMetadata(const std::string& runDir,
                 uint32_t seed,
                 uint64_t run,
                 double distance,
                 uint16_t numberOfUes,
                 uint16_t numberOfEnbs,
                 double simTimeSec,
                 double stopTimeSec,
                 bool useOran,
                 bool useLteHandover,
                 bool useTorch,
                 bool useOnnx,
                 bool useDistance,
                 bool useRsrp,
                 const std::string& dbFileName,
                 const std::string& lateCommandPolicy,
                 const std::string& pathlossModel,
                 double frequencyHz,
                 bool shadowingEnabled,
                 double siteHeight,
                 double ueSpeed,
                 double hysteresisDb,
                 double timeToTriggerMs)
{
    std::ofstream meta(runDir + "/run-info.txt", std::ios::out);
    if (!meta.is_open())
    {
        NS_ABORT_MSG("Cannot open run-info.txt for writing");
    }

    meta << "seed=" << seed << "\n";
    meta << "run=" << run << "\n";
    meta << "distance=" << distance << "\n";
    meta << "numberOfUes=" << numberOfUes << "\n";
    meta << "numberOfEnbs=" << numberOfEnbs << "\n";
    meta << "simTimeSec=" << simTimeSec << "\n";
    meta << "stopTimeSec=" << stopTimeSec << "\n";
    meta << "useOran=" << useOran << "\n";
    meta << "useLteHandover=" << useLteHandover << "\n";
    meta << "useTorch=" << useTorch << "\n";
    meta << "useOnnx=" << useOnnx << "\n";
    meta << "useDistance=" << useDistance << "\n";
    meta << "useRsrp=" << useRsrp << "\n";
    meta << "dbFileName=" << dbFileName << "\n";
    meta << "lateCommandPolicy=" << lateCommandPolicy << "\n";
    meta << "pathlossModel=" << pathlossModel << "\n";
    meta << "frequencyHz=" << frequencyHz << "\n";
    meta << "shadowingEnabled=" << shadowingEnabled << "\n";
    meta << "siteHeight=" << siteHeight << "\n";
    meta << "ueSpeed=" << ueSpeed << "\n";
    meta << "hysteresisDb=" << hysteresisDb << "\n";
    meta << "timeToTriggerMs=" << timeToTriggerMs << "\n";
    meta.close();
}


void
PacketTraceDetailed(Ptr<OutputStreamWrapper> stream,
                    TrafficDirection direction,
                    uint16_t ueId,
                    uint32_t nodeId,
                    Ptr<LteUeNetDevice> ueDev,
                    Ptr<const Packet> p,
                    const Address& /*from*/,
                    const Address& /*to*/)
{
    uint64_t imsi = 0;

    if (ueDev)
    {
        imsi = ueDev->GetImsi();
    }

    *stream->GetStream()
        << Simulator::Now().GetSeconds() << " "
        << ueId << " "
        << nodeId << " "
        << imsi << " "
        << p->GetSize()
        << std::endl;

    if (g_hybridLstmController && imsi != 0)
    {
        g_hybridLstmController->RecordTraffic(imsi, direction, p->GetSize());
    }
}


int
main(int argc, char* argv[])
{

    uint32_t seed = 12345;
    uint64_t run = 1;
    std::string outputRoot = "results";
    std::string runDir;

    std::string pathlossModel = "ns3::ThreeGppUmaPropagationLossModel";
    double frequencyHz = 2.1e9;
    bool shadowingEnabled = true;
    double siteHeight = 25.0;
    double ueSpeed = 10.0;
    double hysteresisDb = 3.0;
    double timeToTriggerMs = 256.0;

    bool useOran = true;
    bool useLteHandover = true;
    bool useTorch = false;
    bool useOnnx = false;
    bool useDistance = false;
    bool useRsrp = false;
    bool enableLstmController = true;

    double lstmDecisionIntervalSec = 1.0;
    uint32_t lstmSeqLen = 32;
    double lstmMinConfidence = 0.55;
    double lstmCooldownSec = 5.0;
    double lstmTriggerThreshold = -1.0;
    double lstmTargetThreshold = -1.0;
    double lstmUtilityThreshold = -1.0;
    uint32_t lstmTargetDistanceTopK = 4;
    bool lstmPreferNonServingTarget = false;
    std::string lstmPythonPath =
        "results_night/.venv/bin/python";
    std::string lstmInferenceScript =
        "results_night/lstm_runtime_infer.py";
    std::string lstmCheckpointPath =
        "results_night/lstm_runs/hex7_v1_seq32_main/best_model.pt";

    bool dbLog = false;
    bool verbose = false;
    bool useAdvancedRicConfig = false;

    double txDelay = 0.001;

    uint16_t numberOfUes = 30;
    uint16_t numberOfSites = 7;
    uint16_t numberOfEnbs = numberOfSites * 3; // 21 sector cells

    Time simTime = Seconds(40);
    Time maxWaitTime = Seconds(0.010);
    std::string processingDelayRv = "ns3::UniformRandomVariable[Min=0.0|Max=0.01]";
    double distance = 500; // distance between eNBs
    Time lmQueryInterval = Seconds(1);
    std::string dbFileName = "oran-repository.db";
    std::string lateCommandPolicy = "DROP";
    const std::string defaultDbFileName = dbFileName;
    const Time cleanupTime = Seconds(1);
    const std::string ueSendIntervalRv = "ns3::ConstantRandomVariable[Constant=0.1]";
    const std::string enbSendIntervalRv = "ns3::ConstantRandomVariable[Constant=0.5]";
    const std::string periodicCellInfoTrigger = "ns3::OranReportTriggerPeriodic";

    // Command line arguments
    CommandLine cmd("lte-oran-helper-lstm-hex7.cc");
    cmd.AddValue("seed", "Global RNG seed", seed);
    cmd.AddValue("run", "Independent RNG run number", run);
    cmd.AddValue("outputRoot", "Root directory for simulation outputs", outputRoot);


    cmd.AddValue("useOran", "Enable O-RAN", useOran);
    cmd.AddValue("useLteHandover",
                 "Enable built-in LTE A3 RSRP handover (baseline, without O-RAN handover)",
                 useLteHandover);
    cmd.AddValue("useTorch", "Use Torch-based LM", useTorch);
    cmd.AddValue("useOnnx", "Use ONNX-based LM", useOnnx);
    cmd.AddValue("useDistance", "Use distance-based LM", useDistance);
    cmd.AddValue("useRsrp", "Use RSRP-based LM", useRsrp);
    cmd.AddValue("enableLstmController", "Enable external Python LSTM controller", enableLstmController);
    cmd.AddValue("lstmDecisionIntervalSec",
                 "Interval between LSTM decision cycles",
                 lstmDecisionIntervalSec);
    cmd.AddValue("lstmSeqLen", "Required history length for LSTM inference", lstmSeqLen);
    cmd.AddValue("lstmMinConfidence",
                 "Minimum confidence required before the LSTM can request handover",
                 lstmMinConfidence);
    cmd.AddValue("lstmCooldownSec",
                 "Minimum cooldown after a completed HO before another LSTM request",
                 lstmCooldownSec);
    cmd.AddValue("lstmTriggerThreshold",
                 "Override Python runtime trigger threshold; negative keeps checkpoint default",
                 lstmTriggerThreshold);
    cmd.AddValue("lstmTargetThreshold",
                 "Override Python runtime target threshold; negative keeps checkpoint default",
                 lstmTargetThreshold);
    cmd.AddValue("lstmUtilityThreshold",
                 "Override Python runtime utility threshold; negative keeps checkpoint default",
                 lstmUtilityThreshold);
    cmd.AddValue("lstmTargetDistanceTopK",
                 "Allow target cell only if it is within K nearest eNBs by geometry; 0 disables this filter",
                 lstmTargetDistanceTopK);
    cmd.AddValue("lstmPreferNonServingTarget",
                 "Prefer the strongest non-serving cell during runtime inference",
                 lstmPreferNonServingTarget);
    cmd.AddValue("lstmPythonPath", "Python interpreter used for LSTM inference", lstmPythonPath);
    cmd.AddValue("lstmInferenceScript",
                 "Python script that reads oran-repository.db and emits LSTM decisions",
                 lstmInferenceScript);
    cmd.AddValue("lstmCheckpointPath",
                 "Checkpoint used by the external Python LSTM inference script",
                 lstmCheckpointPath);

    cmd.AddValue("dbLog", "Enable DB query logging", dbLog);
    cmd.AddValue("verbose", "Verbose LM output", verbose);
    cmd.AddValue("txDelay", "RIC/E2 transmission delay", txDelay);
    cmd.AddValue("lmQueryInterval", "RIC query interval (s)", lmQueryInterval);
    cmd.AddValue("dbFileName", "SQLite DB filename", dbFileName);
    cmd.AddValue("processingDelayRv", "LM processing delay RV", processingDelayRv);
    cmd.AddValue("useAdvancedRicConfig", "Enable advanced RIC settings", useAdvancedRicConfig);
    cmd.AddValue("lateCommandPolicy", "Late command policy", lateCommandPolicy);
    cmd.AddValue("db-log", "Enable printing SQL queries results", dbLog);

    cmd.AddValue("max-wait-time", "The maximum amount of time an LM has to run", maxWaitTime);
    cmd.AddValue("processing-delay-rv",
                 "The random variable that represents the LMs processing delay",
                 processingDelayRv);
    cmd.AddValue("lm-query-interval",
                 "The interval at which to query the LM for commands",
                 lmQueryInterval);
    cmd.AddValue("late-command-policy",
                 "The policy to use for handling commands received after the maximum wait time "
                 "(\"DROP\" or \"SAVE\")",
                 lateCommandPolicy);
    cmd.AddValue("sim-time", "The amount of time to simulate", simTime);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    std::ostringstream runName;
    runName << "seed" << seed << "-run" << run;
    runDir = FindNextRunDir(outputRoot, runName.str());
    runDir = ResolvePath(runDir);
    if (dbFileName.empty() || dbFileName == defaultDbFileName)
    {
        dbFileName = runDir + "/oran-repository.db";
    }
    else
    {
        dbFileName = ResolvePath(dbFileName);
    }

    lstmPythonPath = ResolvePath(lstmPythonPath);
    lstmInferenceScript = ResolvePath(lstmInferenceScript);
    lstmCheckpointPath = ResolvePath(lstmCheckpointPath);

    OranNearRtRic::LateCommandPolicy latePolicy = OranNearRtRic::DROP;
    if (lateCommandPolicy == "DROP")
    {
        latePolicy = OranNearRtRic::DROP;
    }
    else if (lateCommandPolicy == "SAVE")
    {
        latePolicy = OranNearRtRic::SAVE;
    }
    else
    {
        NS_ABORT_MSG("lateCommandPolicy must be either DROP or SAVE");
    }

    const Time stopTime = simTime + cleanupTime;

    std::cout << "Output directory: " << runDir << std::endl;
    std::cout << "Seed=" << seed << ", Run=" << run << std::endl;


    NS_ABORT_MSG_IF(enableLstmController && !useOran,
                    "The hybrid LSTM controller requires useOran=true so it can persist features "
                    "into oran-repository.db.");

    NS_ABORT_MSG_IF(useTorch || useOnnx || useDistance || useRsrp,
                    "lte-oran-helper-lstm-hex7.cc uses an external Python LSTM controller and "
                    "keeps the built-in O-RAN handover LMs disabled.");


    Ptr<OutputStreamWrapper> dlTxTraceStream =
        Create<OutputStreamWrapper>(runDir + "/dl-tx.tr", std::ios::out);
    Ptr<OutputStreamWrapper> dlRxTraceStream =
        Create<OutputStreamWrapper>(runDir + "/dl-rx.tr", std::ios::out);
    Ptr<OutputStreamWrapper> ulTxTraceStream =
        Create<OutputStreamWrapper>(runDir + "/ul-tx.tr", std::ios::out);
    Ptr<OutputStreamWrapper> ulRxTraceStream =
        Create<OutputStreamWrapper>(runDir + "/ul-rx.tr", std::ios::out);

    *dlTxTraceStream->GetStream() << "time ueId nodeId imsi bytes" << std::endl;
    *dlRxTraceStream->GetStream() << "time ueId nodeId imsi bytes" << std::endl;
    *ulTxTraceStream->GetStream() << "time ueId nodeId imsi bytes" << std::endl;
    *ulRxTraceStream->GetStream() << "time ueId nodeId imsi bytes" << std::endl;

    LogComponentEnable("OranNearRtRic", (LogLevel)(LOG_PREFIX_TIME | LOG_WARN));



    Ptr<OutputStreamWrapper> ueCellStateTraceStream =
        Create<OutputStreamWrapper>(runDir + "/ue-cell-state.tr", std::ios::out);

    *ueCellStateTraceStream->GetStream()
        << "time ueId nodeId imsi servingCellId rnti" << std::endl;


    // Increase the buffer size to accomodate the application demand
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(1000 * 1024));
    // Disabled to prevent the automatic cell reselection when signal quality is bad.
    Config::SetDefault("ns3::LteUePhy::EnableRlfDetection", BooleanValue(false));
    ConfigureDefaultLteTraceFiles(runDir);

    // Configure the LTE parameters (pathloss, bandwidth, scheduler)
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();

    lteHelper->SetAttribute("PathlossModel", StringValue(pathlossModel));
    lteHelper->SetPathlossModelAttribute("Frequency", DoubleValue(frequencyHz));
    lteHelper->SetPathlossModelAttribute("ShadowingEnabled", BooleanValue(shadowingEnabled));

    lteHelper->SetEnbAntennaModelType("ns3::CosineAntennaModel");
    lteHelper->SetEnbAntennaModelAttribute("HorizontalBeamwidth", DoubleValue(65.0));
    lteHelper->SetEnbAntennaModelAttribute("VerticalBeamwidth", DoubleValue(15.0));
    lteHelper->SetEnbAntennaModelAttribute("MaxGain", DoubleValue(0.0));

    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(50));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(50));
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    lteHelper->SetSchedulerAttribute("HarqEnabled", BooleanValue(true));
    
    if (useLteHandover)
    {
        lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
        lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(hysteresisDb));
        lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(static_cast<uint64_t>(timeToTriggerMs))));
    }
    else
    {
        lteHelper->SetHandoverAlgorithmType("ns3::NoOpHandoverAlgorithm");
    }

    // Deploy the EPC
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // Create a single remote host
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // IP configuration
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(65000));
    p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(0)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    NodeContainer ueNodes;
    NodeContainer enbNodes;
    enbNodes.Create(numberOfEnbs);
    ueNodes.Create(numberOfUes);

    // eNBs must already have a MobilityModel before hexGrid positions them
    MobilityHelper mobilityEnbs;
    mobilityEnbs.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityEnbs.Install(enbNodes);

    // 7 sites x 3 sectors = 21 eNB
    Ptr<LteHexGridEnbTopologyHelper> hexGrid = CreateObject<LteHexGridEnbTopologyHelper>();
    hexGrid->SetLteHelper(lteHelper);

    hexGrid->SetAttribute("InterSiteDistance", DoubleValue(distance));
    hexGrid->SetAttribute("SiteHeight", DoubleValue(siteHeight));
    hexGrid->SetAttribute("SectorOffset", DoubleValue(0.5));
    hexGrid->SetAttribute("GridWidth", UintegerValue(2));
    hexGrid->SetAttribute("MinX", DoubleValue(-distance));
    hexGrid->SetAttribute("MinY", DoubleValue(-distance));

    NetDeviceContainer enbLteDevs = hexGrid->SetPositionAndInstallEnbDevice(enbNodes);

    MobilityHelper mobilityUes;

    // случайное начальное размещение
    Ptr<UniformRandomVariable> xRv = CreateObject<UniformRandomVariable>();
    xRv->SetAttribute("Min", DoubleValue(-distance*2));
    xRv->SetAttribute("Max", DoubleValue(distance*2));

    Ptr<UniformRandomVariable> yRv = CreateObject<UniformRandomVariable>();
    yRv->SetAttribute("Min", DoubleValue(-distance*2));
    yRv->SetAttribute("Max", DoubleValue(500));

    Ptr<ListPositionAllocator> positionAllocUes = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        positionAllocUes->Add(Vector(xRv->GetValue(), yRv->GetValue(), 1.5));
    }
    mobilityUes.SetPositionAllocator(positionAllocUes);

    // движение через Random Walk
    mobilityUes.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(-distance*2.1, distance*2.1, -distance*2.1, distance*2.1)),
        "Mode", StringValue("Time"),
        "Time", TimeValue(Seconds(1.0)),
        "Speed", StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(ueSpeed) + "]"),
        "Direction", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=6.283185307]")
    );

    mobilityUes.Install(ueNodes);

    // Install LTE Devices to the nodes
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);



    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));
    // Assign IP address to UEs, and install applications
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<Node> ueNode = ueNodes.Get(u);
        // Set the default gateway for the UE
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    lteHelper->AttachToClosestEnb(ueLteDevs, enbLteDevs);

    lteHelper->AddX2Interface(enbNodes);


    // Install and start applications on UEs and remote host
    uint16_t basePort = 1000;
    ApplicationContainer remoteApps;
    ApplicationContainer ueApps;
    ApplicationContainer ulSinks;
    ApplicationContainer ulApps;

    Ptr<RandomVariableStream> onTimeRv = CreateObject<UniformRandomVariable>();
    onTimeRv->SetAttribute("Min", DoubleValue(1.0));
    onTimeRv->SetAttribute("Max", DoubleValue(5.0));
    Ptr<RandomVariableStream> offTimeRv = CreateObject<UniformRandomVariable>();
    offTimeRv->SetAttribute("Min", DoubleValue(1.0));
    offTimeRv->SetAttribute("Max", DoubleValue(5.0));

    for (uint16_t i = 0; i < ueNodes.GetN(); i++)
    {
        uint16_t port = basePort * (i + 1);

        Ptr<LteUeNetDevice> ueDev = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        uint32_t nodeId = ueNodes.Get(i)->GetId();

        PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), port));
        ueApps.Add(dlPacketSinkHelper.Install(ueNodes.Get(i)));
        // Enable the tracing of RX packets
        ueApps.Get(i)->TraceConnectWithoutContext(
                "RxWithAddresses",
                MakeBoundCallback(&PacketTraceDetailed,
                      dlRxTraceStream,
                      TrafficDirection::DL_RX,
                      i,
                      nodeId,
                      ueDev));


        Ptr<OnOffApplication> streamingServer = CreateObject<OnOffApplication>();
        remoteApps.Add(streamingServer);
        // Attributes
        streamingServer->SetAttribute(
            "Remote",
            AddressValue(InetSocketAddress(ueIpIface.GetAddress(i), port)));
        streamingServer->SetAttribute("DataRate", DataRateValue(DataRate("3000000bps")));
        streamingServer->SetAttribute("PacketSize", UintegerValue(1500));
        streamingServer->SetAttribute("OnTime", PointerValue(onTimeRv));
        streamingServer->SetAttribute("OffTime", PointerValue(offTimeRv));

        remoteHost->AddApplication(streamingServer);
        streamingServer->TraceConnectWithoutContext(
            "TxWithAddresses",
            MakeBoundCallback(&PacketTraceDetailed,
                      dlTxTraceStream,
                      TrafficDirection::DL_TX,
                      i,
                      nodeId,
                      ueDev));


        uint16_t ulPort = 50000 + i;

        PacketSinkHelper ulPacketSinkHelper("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), ulPort));
        ulSinks.Add(ulPacketSinkHelper.Install(remoteHost));

        ulSinks.Get(i)->TraceConnectWithoutContext(
            "RxWithAddresses",
            MakeBoundCallback(&PacketTraceDetailed,
                      ulRxTraceStream,
                      TrafficDirection::UL_RX,
                      i,
                      nodeId,
                      ueDev));

        Ptr<OnOffApplication> ulClient = CreateObject<OnOffApplication>();
        ulApps.Add(ulClient);

        ulClient->SetAttribute(
            "Remote",
            AddressValue(InetSocketAddress(internetIpIfaces.GetAddress(1), ulPort)));
        ulClient->SetAttribute("DataRate", DataRateValue(DataRate("1000000bps")));
        ulClient->SetAttribute("PacketSize", UintegerValue(512));
        ulClient->SetAttribute("OnTime", PointerValue(onTimeRv));
        ulClient->SetAttribute("OffTime", PointerValue(offTimeRv));

        ueNodes.Get(i)->AddApplication(ulClient);

        ulClient->TraceConnectWithoutContext(
            "TxWithAddresses",
            MakeBoundCallback(&PacketTraceDetailed,
                      ulTxTraceStream,
                      TrafficDirection::UL_TX,
                      i,
                      nodeId,
                      ueDev));
    }

    // Inidcate when to start streaming
    remoteApps.Start(Seconds(2));
    // Indicate when to stop streaming
    remoteApps.Stop(simTime);

    // UE applications start listening
    ueApps.Start(Seconds(1));
    // UE applications stop listening
    ueApps.Stop(stopTime);


    ulSinks.Start(Seconds(1.0));
    ulSinks.Stop(stopTime);

    ulApps.Start(Seconds(2.0));
    ulApps.Stop(simTime);

    // ==========================
    // ORAN BEGIN (via OranHelper)
    // ==========================
    if (useOran)
    {
        if (!dbFileName.empty())
        {
            std::remove(dbFileName.c_str());
        }

        Ptr<OranNearRtRic> nearRtRic = nullptr;
        OranE2NodeTerminatorContainer e2NodeTerminatorsUes;
        OranE2NodeTerminatorContainer e2NodeTerminatorsEnbs;

        Ptr<OranHelper> oranHelper = CreateObject<OranHelper>();

        // --------------------------------------------------
        // Common helper configuration
        // --------------------------------------------------
        oranHelper->SetAttribute("Verbose", BooleanValue(verbose));
        oranHelper->SetAttribute("LmQueryInterval", TimeValue(lmQueryInterval));
        oranHelper->SetAttribute("LmQueryLateCommandPolicy", EnumValue(latePolicy));
        oranHelper->SetAttribute(
            "RicTransmissionDelayRv",
            StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(txDelay) + "]"));

        if (useAdvancedRicConfig)
        {
            oranHelper->SetAttribute("LmQueryMaxWaitTime", TimeValue(maxWaitTime));
            oranHelper->SetAttribute("E2NodeInactivityThreshold", TimeValue(Seconds(2)));
            oranHelper->SetAttribute(
                "E2NodeInactivityIntervalRv",
                StringValue("ns3::ConstantRandomVariable[Constant=2]"));
        }

        // --------------------------------------------------
        // Data repository
        // --------------------------------------------------
        oranHelper->SetDataRepository("ns3::OranDataRepositorySqlite",
                                      "DatabaseFile",
                                      StringValue(dbFileName));

        // --------------------------------------------------
        // Select default Logic Module
        // --------------------------------------------------
        std::string lmType = "ns3::OranLmNoop";

        TypeId selectedLmTid;
        NS_ABORT_MSG_IF(!TypeId::LookupByNameFailSafe(lmType, &selectedLmTid),
                        "Requested logic module is not available: " + lmType);

        if (!processingDelayRv.empty())
        {
            oranHelper->SetDefaultLogicModule(lmType,
                                              "ProcessingDelayRv",
                                              StringValue(processingDelayRv));
        }
        else
        {
            oranHelper->SetDefaultLogicModule(
                lmType,
                "ProcessingDelayRv",
                StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        }

        oranHelper->SetConflictMitigationModule("ns3::OranCmmNoop");

        // --------------------------------------------------
        // Create Near-RT RIC
        // --------------------------------------------------
        nearRtRic = oranHelper->CreateNearRtRic();

        // DB logging to terminal
        if (dbLog)
        {
            nearRtRic->Data()->TraceConnectWithoutContext("QueryRc",
                                                          MakeCallback(&QueryRcSink));
        }

        // --------------------------------------------------
        // UE-side terminators + built-in reporters
        // --------------------------------------------------
        oranHelper->SetE2NodeTerminator(
            "ns3::OranE2NodeTerminatorLteUe",
            "RegistrationIntervalRv",
            StringValue("ns3::ConstantRandomVariable[Constant=1]"),
            "SendIntervalRv",
            StringValue(ueSendIntervalRv),
            "TransmissionDelayRv",
            StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(txDelay) + "]"));

        oranHelper->AddReporter("ns3::OranReporterLocation",
                                "Trigger",
                                StringValue("ns3::OranReportTriggerPeriodic"));

        oranHelper->AddReporter("ns3::OranReporterLteUeCellInfo",
                                "Trigger",
                                StringValue("ns3::OranReportTriggerLteUeHandover[InitialReport=true]"));

        oranHelper->AddReporter("ns3::OranReporterLteUeCellInfo",
                                "Trigger",
                                StringValue(periodicCellInfoTrigger));

        e2NodeTerminatorsUes.Add(oranHelper->DeployTerminators(nearRtRic, ueNodes));

        // --------------------------------------------------
        // eNB-side terminators + built-in reporters
        // --------------------------------------------------
        oranHelper->SetE2NodeTerminator(
            "ns3::OranE2NodeTerminatorLteEnb",
            "RegistrationIntervalRv",
            StringValue("ns3::ConstantRandomVariable[Constant=1]"),
            "SendIntervalRv",
            StringValue(enbSendIntervalRv),
            "TransmissionDelayRv",
            StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(txDelay) + "]"));

        oranHelper->AddReporter("ns3::OranReporterLocation",
                                "Trigger",
                                StringValue("ns3::OranReportTriggerPeriodic"));

        e2NodeTerminatorsEnbs.Add(oranHelper->DeployTerminators(nearRtRic, enbNodes));

        // --------------------------------------------------
        // Activation
        // --------------------------------------------------
        Simulator::Schedule(Seconds(1.0),
                            &OranHelper::ActivateAndStartNearRtRic,
                            oranHelper,
                            nearRtRic);

        Simulator::Schedule(Seconds(1.5),
                            &OranHelper::ActivateE2NodeTerminators,
                            oranHelper,
                            e2NodeTerminatorsEnbs);

        Simulator::Schedule(Seconds(2.0),
                            &OranHelper::ActivateE2NodeTerminators,
                            oranHelper,
                            e2NodeTerminatorsUes);

        // --------------------------------------------------
        // IMPORTANT:
        // Keep custom trace-based reporters manually if needed
        // (AppLoss / direct PHY RSRP-RSRQ hook)
        // --------------------------------------------------
        for (uint32_t idx = 0; idx < ueNodes.GetN(); idx++)
        {
            Ptr<Node> ueNode = ueNodes.Get(idx);
            Ptr<LteUeNetDevice> lteUeDevice = ueLteDevs.Get(idx)->GetObject<LteUeNetDevice>();

            // Custom AppLoss reporter still manual
            Ptr<OranReporterAppLoss> appLossReporter = CreateObject<OranReporterAppLoss>();
            Ptr<OranReporterLteUeRsrpRsrq> rsrpRsrqReporter =
                CreateObject<OranReporterLteUeRsrpRsrq>();

            // Bind to deployed UE terminator
            Ptr<OranE2NodeTerminator> baseTerm =
                e2NodeTerminatorsUes.Get(idx);
            Ptr<OranE2NodeTerminatorLteUe> ueTerm =
                DynamicCast<OranE2NodeTerminatorLteUe>(baseTerm);

            if (ueTerm)
            {
                appLossReporter->SetAttribute("Terminator", PointerValue(ueTerm));
                rsrpRsrqReporter->SetAttribute("Terminator", PointerValue(ueTerm));
                ueTerm->AddReporter(appLossReporter);
                ueTerm->AddReporter(rsrpRsrqReporter);

                if (idx < remoteApps.GetN() && idx < ueApps.GetN())
                {
                    remoteApps.Get(idx)->TraceConnectWithoutContext(
                        "Tx",
                        MakeCallback(&ns3::OranReporterAppLoss::AddTx, appLossReporter));

                    ueApps.Get(idx)->TraceConnectWithoutContext(
                        "Rx",
                        MakeCallback(&ns3::OranReporterAppLoss::AddRx, appLossReporter));
                }

                if (lteUeDevice)
                {
                    Ptr<LteUePhy> uePhy = lteUeDevice->GetPhy();
                    if (uePhy)
                    {
                        uePhy->TraceConnectWithoutContext(
                            "ReportUeMeasurements",
                            MakeCallback(&ns3::OranReporterLteUeRsrpRsrq::ReportRsrpRsrq,
                                         rsrpRsrqReporter));

                        uePhy->TraceConnectWithoutContext(
                            "ReportUeMeasurements",
                            MakeBoundCallback(&TraceUeMeasurementsRsrpRsrq,
                                              lteUeDevice->GetImsi()));
                    }
                }
            }
        }
    }
    // ========================
    // ORAN END (via OranHelper)
    // ========================

    g_hybridLstmController = std::make_unique<HybridLstmController>(dbFileName,
                                                                    runDir,
                                                                    lteHelper,
                                                                    ueLteDevs,
                                                                    enbLteDevs,
                                                                    enableLstmController,
                                                                    lstmDecisionIntervalSec,
                                                                    lstmSeqLen,
                                                                    lstmMinConfidence,
                                                                    lstmCooldownSec,
                                                                    lstmTriggerThreshold,
                                                                    lstmTargetThreshold,
                                                                    lstmUtilityThreshold,
                                                                    lstmTargetDistanceTopK,
                                                                    lstmPreferNonServingTarget,
                                                                    lstmPythonPath,
                                                                    lstmInferenceScript,
                                                                    lstmCheckpointPath);
    g_hybridLstmController->Initialize();
    g_hybridLstmController->Start();


    // Trace successful handovers
    Ptr<OutputStreamWrapper> handoverStartTraceStream =
    Create<OutputStreamWrapper>(runDir + "/handover-start.tr", std::ios::out);
    Ptr<OutputStreamWrapper> handoverEndTraceStream =
        Create<OutputStreamWrapper>(runDir + "/handover-end.tr", std::ios::out);
    Ptr<OutputStreamWrapper> handoverStatsTraceStream =
        Create<OutputStreamWrapper>(runDir + "/handover-stats.tr", std::ios::out);

    *handoverStartTraceStream->GetStream() << "time imsi sourceCellId targetCellId rnti" << std::endl;
    *handoverEndTraceStream->GetStream() << "time imsi targetCellId rnti successfulHoCount pingPongCount isPingPong" << std::endl;
    *handoverStatsTraceStream->GetStream() << "time imsi successfulHoCount pingPongCount pingPongRate" << std::endl;

    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                              MakeBoundCallback(&HandoverStartTrace, handoverStartTraceStream));

    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                              MakeBoundCallback(&HandoverEndTraceDetailed,
                                                handoverEndTraceStream,
                                                handoverStatsTraceStream));


    // Periodically trace node positions
    Ptr<OutputStreamWrapper> positionTraceStream =
    Create<OutputStreamWrapper>(runDir + "/positions.tr", std::ios::out);
    *positionTraceStream->GetStream() << "time ueId nodeId x y z" << std::endl;
    Simulator::Schedule(Seconds(1), &PositionTrace, positionTraceStream, ueNodes);
    
    // Trace current-cell RSRP and SINR
    Ptr<OutputStreamWrapper> rsrpRsrqSinrTraceStream =
    Create<OutputStreamWrapper>(runDir + "/rsrp-sinr.tr", std::ios::out);
    *rsrpRsrqSinrTraceStream->GetStream() << "time rnti cellId rsrp sinr ccId" << std::endl;

    for (NetDeviceContainer::Iterator it = ueLteDevs.Begin(); it != ueLteDevs.End(); ++it)
    {
        Ptr<NetDevice> device = *it;
        Ptr<LteUeNetDevice> lteUeDevice = device->GetObject<LteUeNetDevice>();
        if (lteUeDevice)
        {
            Ptr<LteUePhy> uePhy = lteUeDevice->GetPhy();
            uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr",
                                              MakeBoundCallback(&TraceCurrentCellRsrpSinr,
                                                                rsrpRsrqSinrTraceStream,
                                                                lteUeDevice->GetImsi()));
        }
    }
    
    Simulator::Schedule(stopTime - MilliSeconds(1),
                        &SaveEnbPositionsToFile,
                        enbNodes,
                        enbLteDevs,
                        runDir + "/location_end.txt");

    WriteRunMetadata(runDir,
                 seed,
                 run,
                 distance,
                 numberOfUes,
                 numberOfEnbs,
                 simTime.GetSeconds(),
                 stopTime.GetSeconds(),
                 useOran,
                 useLteHandover,
                 useTorch,
                 useOnnx,
                 useDistance,
                 useRsrp,
                 dbFileName,
                 lateCommandPolicy,
                 pathlossModel,
                 frequencyHz,
                 shadowingEnabled,
                 siteHeight,
                 ueSpeed,
                 hysteresisDb,
                 timeToTriggerMs);

    {
        std::ofstream metaAppend(runDir + "/run-info.txt", std::ios::app);
        metaAppend << "handoverScenario=HYBRID_LSTM_A3_FALLBACK\n";
        metaAppend << "a3SafetyNetEnabled=" << useLteHandover << "\n";
        metaAppend << "enableLstmController=" << enableLstmController << "\n";
        metaAppend << "lstmDecisionIntervalSec=" << lstmDecisionIntervalSec << "\n";
        metaAppend << "lstmSeqLen=" << lstmSeqLen << "\n";
        metaAppend << "lstmMinConfidence=" << lstmMinConfidence << "\n";
        metaAppend << "lstmCooldownSec=" << lstmCooldownSec << "\n";
        metaAppend << "lstmTriggerThreshold=" << lstmTriggerThreshold << "\n";
        metaAppend << "lstmTargetThreshold=" << lstmTargetThreshold << "\n";
        metaAppend << "lstmUtilityThreshold=" << lstmUtilityThreshold << "\n";
        metaAppend << "lstmTargetDistanceTopK=" << lstmTargetDistanceTopK << "\n";
        metaAppend << "lstmPreferNonServingTarget=" << lstmPreferNonServingTarget << "\n";
        metaAppend << "lstmPythonPath=" << lstmPythonPath << "\n";
        metaAppend << "lstmInferenceScript=" << lstmInferenceScript << "\n";
        metaAppend << "lstmCheckpointPath=" << lstmCheckpointPath << "\n";
    }

    Simulator::Schedule(Seconds(1.0),
                    &UeCellStateTrace,
                    ueCellStateTraceStream,
                    ueLteDevs,
                    ueNodes);


    /* Enabling Tracing for the simulation scenario */
    lteHelper->EnablePhyTraces();
    lteHelper->EnableMacTraces();
    lteHelper->EnableRlcTraces();
    lteHelper->EnablePdcpTraces();

    Simulator::Stop(stopTime);
    Simulator::Run();

    Simulator::Destroy();

    return 0;
}
