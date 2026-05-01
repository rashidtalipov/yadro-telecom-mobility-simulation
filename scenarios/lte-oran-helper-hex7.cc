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
#include <sstream>

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

NS_LOG_COMPONENT_DEFINE("LteOranHexGridBaseline");

/**
 * LTE + O-RAN hex-grid baseline scenario.
 *
 * 7 LTE sites with 3 sectors each (21 cells) serve 30 mobile UEs.
 * Built-in LTE A3 RSRP handover is used as the mobility baseline.
 * When O-RAN is enabled, the Near-RT RIC collects UE location,
 * serving-cell information, and application-loss statistics through
 * E2 node terminators and stores them in the SQLite data repository.
 */

#include <vector>
#include <unordered_map>
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
MoveDefaultLteTraceFiles(const std::string& runDir)
{
    namespace fs = std::filesystem;

    // SQLite output is already written directly to dbFileName in the run directory.
    std::vector<std::string> files = {
        "UlSinrStats.txt",
        "UlTxPhyStats.txt",
        "UlRxPhyStats.txt",
        "UlMacStats.txt",
        "UlRlcStats.txt",
        "UlPdcpStats.txt",
        "DlRsrpSinrStats.txt",
        "DlTxPhyStats.txt",
        "DlRxPhyStats.txt",
        "DlMacStats.txt",
        "DlRsrpSinrStats.txt",
        "DlMacStats.txt",
        "DlRlcStats.txt",
        "DlPdcpStats.txt",
    };

    for (const auto& f : files)
    {
        if (fs::exists(f))
        {
            fs::rename(f, runDir + "/" + f);
        }
    }
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
    // bool useLstm = false;   // later

    bool dbLog = false;
    bool verbose = false;
    bool useAdvancedRicConfig = false;

    double txDelay = 0.001;

    uint16_t numberOfUes = 30;
    uint16_t numberOfSites = 7;
    uint16_t numberOfEnbs = numberOfSites * 3; // 21 sector cells

    Time simTime = Seconds(10);
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
    CommandLine cmd(__FILE__);
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
    // cmd.AddValue("useLstm", "Use LSTM-based LM", useLstm);

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
    if (dbFileName.empty() || dbFileName == defaultDbFileName)
    {
        dbFileName = runDir + "/oran-repository.db";
    }

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


    if (useOran && !useLteHandover)
    {
        const uint32_t enabledLmCount = static_cast<uint32_t>(useTorch) +
                                        static_cast<uint32_t>(useOnnx) +
                                        static_cast<uint32_t>(useDistance) +
                                        static_cast<uint32_t>(useRsrp);
        NS_ABORT_MSG_IF(enabledLmCount != 1,
                        "Exactly one of useTorch/useOnnx/useDistance/useRsrp must be true when "
                        "O-RAN controls handover.");
    }

    if (useOran && useLteHandover)
    {
        const uint32_t enabledLmCount = static_cast<uint32_t>(useTorch) +
                                        static_cast<uint32_t>(useOnnx) +
                                        static_cast<uint32_t>(useDistance) +
                                        static_cast<uint32_t>(useRsrp);
        NS_ABORT_MSG_IF(enabledLmCount != 0,
                        "When LTE handover is enabled, O-RAN must be observation-only "
                        "(all logic-module handover flags must be false).");
    }


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

        if (useOnnx)
        {
            lmType = "ns3::OranLmLte2LteOnnxHandover";
        }
        else if (useTorch)
        {
            lmType = "ns3::OranLmLte2LteTorchHandover";
        }
        else if (useDistance)
        {
            lmType = "ns3::OranLmLte2LteDistanceHandover";
        }
        else if (useRsrp)
        {
            lmType = "ns3::OranLmLte2LteRsrpHandover";
        }

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

            // Custom AppLoss reporter still manual
            Ptr<OranReporterAppLoss> appLossReporter = CreateObject<OranReporterAppLoss>();

            // Bind to deployed UE terminator
            Ptr<OranE2NodeTerminator> baseTerm =
                e2NodeTerminatorsUes.Get(idx);
            Ptr<OranE2NodeTerminatorLteUe> ueTerm =
                DynamicCast<OranE2NodeTerminatorLteUe>(baseTerm);

            if (ueTerm)
            {
                appLossReporter->SetAttribute("Terminator", PointerValue(ueTerm));
                ueTerm->AddReporter(appLossReporter);

                if (idx < remoteApps.GetN() && idx < ueApps.GetN())
                {
                    remoteApps.Get(idx)->TraceConnectWithoutContext(
                        "Tx",
                        MakeCallback(&ns3::OranReporterAppLoss::AddTx, appLossReporter));

                    ueApps.Get(idx)->TraceConnectWithoutContext(
                        "Rx",
                        MakeCallback(&ns3::OranReporterAppLoss::AddRx, appLossReporter));
                }
            }
        }
    }
    // ========================
    // ORAN END (via OranHelper)
    // ========================


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
            uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr", MakeBoundCallback(&TraceCurrentCellRsrpSinr, rsrpRsrqSinrTraceStream));
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
    
    MoveDefaultLteTraceFiles(runDir);

    return 0;
}
