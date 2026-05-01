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


#include "ns3/oran-reporter-apploss.h"
#include "ns3/oran-helper.h"
#include "ns3/oran-near-rt-ric.h"
#include "ns3/oran-data-repository-sqlite.h"
#include "ns3/oran-cmm-handover.h"
#include "ns3/oran-lm.h"
#include "ns3/oran-reporter-location.h"
#include "ns3/oran-reporter-lte-ue-cell-info.h"
#include "ns3/oran-reporter-lte-ue-rsrp-rsrq.h"
#include "ns3/oran-e2-node-terminator-lte-ue.h"
#include "ns3/oran-e2-node-terminator-lte-enb.h"

#include "ns3/lte-ue-net-device.h"
#include "ns3/lte-ue-phy.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OranLte2LteRsrpHandoverExample");
/**
 * ORAN handover example.
 *
 * This example demonstrates how the use of a custom LM query trigger in the
 * Near-RT RIC can trigger a query outside of the periodic query interval
 * based on a report and the information that it contains that was received
 * by the Near-RT RIC.
 */

/**
 * Usage example of the ORAN models, configured with the ORAN Helper.
 *
 * The scenario consists of an LTE UE moving back and forth
 * between 2 LTE eNBs. The LTE UE reports to the RIC its location
 * and current Cell ID. In the RIC, an LM will periodically check
 * the position, and if needed, issue a handover command.
 */

void
TraceEnbRx(std::string context, uint16_t rnti, uint8_t lcid, uint32_t bytes, uint64_t delay)
{
    std::cout << Simulator::Now().GetSeconds() << " s: " << context << " recieved " << bytes
              << " bytes from RNTI " << (uint32_t)rnti << std::endl;
}

void
NotifyHandoverEndOkEnb(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << " s:" << context << " eNB CellId " << cellid
              << ": completed handover of UE with IMSI " << imsi << " RNTI " << rnti << std::endl;

    Config::Disconnect("NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteEnbNetDevice/LteEnbRrc/"
                       "UeMap/*/DataRadioBearerMap/*/LteRlc/RxPDU",
                       MakeCallback(&TraceEnbRx));

    Config::Connect("NodeList/*/DeviceList/*/$ns3::LteNetDevice/$ns3::LteEnbNetDevice/LteEnbRrc/"
                    "UeMap/*/DataRadioBearerMap/*/LteRlc/RxPDU",
                    MakeCallback(&TraceEnbRx));
}


/**
 * ORAN handover example. Based on the LTE module's "lena-x2-handover.cc".
 */


/**
 * Example of the ORAN models.
 *
 * The scenario consists of an LTE UE moving back and forth
 * between 2 LTE eNBs. The LTE UE reports its location to the RIC
 * and current Cell ID. In the RIC, an LM will periodically check
 * the RSRP and RSRQ of UE, and if needed, issue a handover command.
 *
 * This example demonstrates how to configure processing delays for the LMs.
 */

void
SaveEnbPositionsToFile(NodeContainer enbNodes)
{
    std::ofstream out("location_end.txt", std::ios::out);

    if (!out.is_open())
    {
        std::cerr << "Cannot open location_end.txt" << std::endl;
        return;
    }

    out << "time,enbId,x,y,z" << std::endl;

    double now = Simulator::Now().GetSeconds();

    for (uint32_t i = 0; i < enbNodes.GetN(); ++i)
    {
        Ptr<MobilityModel> mob = enbNodes.Get(i)->GetObject<MobilityModel>();
        if (mob)
        {
            Vector p = mob->GetPosition();
            out << now << "," << i << "," << p.x << "," << p.y << "," << p.z << std::endl;
        }
        else
        {
            out << now << "," << i << ",NO_MOBILITY,NO_MOBILITY,NO_MOBILITY" << std::endl;
        }
    }

    out.close();
}


 // Tracing rsrp, rsrq, and sinr
// ns-3 >= 3.28
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


// Trace RX'd packets
void
RxTrace(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> p, const Address& from, const Address& to)
{
    uint16_t ueId = (InetSocketAddress::ConvertFrom(to).GetPort() / 1000);

    *stream->GetStream()
      << Simulator::Now().GetSeconds() << " "
      << ueId
      << " RX "
      << p->GetSize()
      << std::endl;
}

// Trace TX'd packets
void
TxTrace(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> p, const Address& from, const Address& to)
{
    uint16_t ueId = (InetSocketAddress::ConvertFrom(to).GetPort() / 1000);

    *stream->GetStream()
      << Simulator::Now().GetSeconds() << " "
      << ueId
      << " TX "
      << p->GetSize()
      << std::endl;
}

// Trace each node's location
void
PositionTrace(Ptr<OutputStreamWrapper> stream, NodeContainer nodes)
{
    for (uint32_t i = 0; i < nodes.GetN(); i++)
    {
        Vector pos = nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
        *stream->GetStream()
            << Simulator::Now().GetSeconds() << " "
            << nodes.Get(i)->GetId() << " "
            << pos.x << " "
            << pos.y
            << std::endl;
    }

    Simulator::Schedule(Seconds(1), &PositionTrace, stream, nodes);
}

// Trace handover events
void
HandoverTrace(Ptr<OutputStreamWrapper> stream, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
    *stream->GetStream()
        << Simulator::Now().GetSeconds() << " "
        << imsi << " "
        << cellid << " "
        << rnti
        << std::endl;
}

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

// Function to change node velocities
void
ReverseVelocity(NodeContainer nodes, Time interval)
{
    for (uint32_t idx = 0; idx < nodes.GetN(); idx++)
    {
        Ptr<ConstantVelocityMobilityModel> mobility =
            nodes.Get(idx)->GetObject<ConstantVelocityMobilityModel>();
        mobility->SetVelocity(Vector(mobility->GetVelocity().x * -1, 0, 0));
    }

    Simulator::Schedule(interval, &ReverseVelocity, nodes, interval);
}

int
main(int argc, char* argv[])
{

    bool useOran = true;
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
    uint16_t numberOfEnbs = 7;
    Time simTime = Seconds(300);
    Time maxWaitTime = Seconds(0.010); 
    std::string processingDelayRv = "ns3::NormalRandomVariable[Mean=0.005|Variance=0.000031]";
    double distance = 400; // distance between eNBs
    Time interval = Seconds(15);

    Time lmQueryInterval = Seconds(1);
    std::string dbFileName = "oran-repository.db";
    std::string lateCommandPolicy = "DROP";

    // Command line arguments
    CommandLine cmd(__FILE__);
    cmd.AddValue("useOran", "Enable O-RAN", useOran);
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

    LogComponentEnable("OranNearRtRic", (LogLevel)(LOG_PREFIX_TIME | LOG_WARN));

    // Increase the buffer size to accomodate the application demand
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(1000 * 1024));
    // Disabled to prevent the automatic cell reselection when signal quality is bad.
    Config::SetDefault("ns3::LteUePhy::EnableRlfDetection", BooleanValue(false));

    // Configure the LTE parameters (pathloss, bandwidth, scheduler)
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    lteHelper->SetAttribute("PathlossModel", StringValue("ns3::Cost231PropagationLossModel"));
    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(50));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(50));
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    lteHelper->SetSchedulerAttribute("HarqEnabled", BooleanValue(true));
    lteHelper->SetHandoverAlgorithmType("ns3::NoOpHandoverAlgorithm"); // disable automatic handover
    
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

    // Create eNB and UE
    NodeContainer ueNodes;
    NodeContainer enbNodes;
    enbNodes.Create(numberOfEnbs);
    ueNodes.Create(numberOfUes);

    // Install Mobility Model for eNB (Constant Position at (0, 0, 0)
    Ptr<ListPositionAllocator> positionAllocEnbs = CreateObject<ListPositionAllocator>();

    double R = distance;          // радиус кольца
    positionAllocEnbs->Add(Vector(0, 0, 20)); // центр

    for (int k = 0; k < 6; ++k)
    {
        double ang = (M_PI / 3.0) * k; // 0,60,120..300 градусов
        positionAllocEnbs->Add(Vector(R * std::cos(ang), R * std::sin(ang), 20));
    }

    // Install Mobility Model for eNB (Constant Positions)
    MobilityHelper mobilityEnbs;
    mobilityEnbs.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityEnbs.SetPositionAllocator(positionAllocEnbs);
    mobilityEnbs.Install(enbNodes);

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
        positionAllocUes->Add(Vector(xRv->GetValue(), yRv->GetValue(), 1.0));
    }
    mobilityUes.SetPositionAllocator(positionAllocUes);

    // движение через Random Walk
    mobilityUes.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(-distance*2.1, distance*2.1, -distance*2.1, distance*2.1)),
        "Mode", StringValue("Time"),
        "Time", TimeValue(Seconds(1.0)),
        "Speed", StringValue("ns3::ConstantRandomVariable[Constant=2]"),
        "Direction", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=6.283185307]")
    );

    mobilityUes.Install(ueNodes);

    // Install LTE Devices to the nodes
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
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

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<MobilityModel> ueMob = ueNodes.Get(u)->GetObject<MobilityModel>();

        double bestDist = 1e18;
        uint32_t bestEnb = 0;

        for (uint32_t b = 0; b < enbNodes.GetN(); ++b)
        {
            Ptr<MobilityModel> enbMob = enbNodes.Get(b)->GetObject<MobilityModel>();
            double d = ueMob->GetDistanceFrom(enbMob);
            if (d < bestDist)
            {
                bestDist = d;
                bestEnb = b;
            }
        }

        lteHelper->Attach(ueLteDevs.Get(u), enbLteDevs.Get(bestEnb));
    }

    lteHelper->AddX2Interface(enbNodes);


    // Install and start applications on UEs and remote host
    uint16_t basePort = 1000;
    ApplicationContainer remoteApps;
    ApplicationContainer ueApps;

    Ptr<RandomVariableStream> onTimeRv = CreateObject<UniformRandomVariable>();
    onTimeRv->SetAttribute("Min", DoubleValue(1.0));
    onTimeRv->SetAttribute("Max", DoubleValue(5.0));
    Ptr<RandomVariableStream> offTimeRv = CreateObject<UniformRandomVariable>();
    offTimeRv->SetAttribute("Min", DoubleValue(1.0));
    offTimeRv->SetAttribute("Max", DoubleValue(5.0));

    Ptr<OutputStreamWrapper> packetTraceStream = Create<OutputStreamWrapper>("packet.tr", std::ios::out);

    for (uint16_t i = 0; i < ueNodes.GetN(); i++)
    {
        uint16_t port = basePort * (i + 1);

        PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                            InetSocketAddress(Ipv4Address::GetAny(), port));
        ueApps.Add(dlPacketSinkHelper.Install(ueNodes.Get(i)));
        // Enable the tracing of RX packets
        ueApps.Get(i)->TraceConnectWithoutContext("RxWithAddresses", MakeBoundCallback(&RxTrace, packetTraceStream));

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
        streamingServer->TraceConnectWithoutContext("TxWithAddresses", MakeBoundCallback(&TxTrace, packetTraceStream));
    }

    // Inidcate when to start streaming
    remoteApps.Start(Seconds(2));
    // Indicate when to stop streaming
    remoteApps.Stop(simTime + Seconds(10));

    // UE applications start listening
    ueApps.Start(Seconds(1));
    // UE applications stop listening
    ueApps.Stop(simTime + Seconds(15));


    // ==========================
    // ORAN BEGIN (Unified Block)
    // ==========================
    if (useOran)
    {
        if (!dbFileName.empty())
        {
            std::remove(dbFileName.c_str());
        }

        // --------------------------------------------------
        // Select default Logic Module
        // --------------------------------------------------
        TypeId defaultLmTid = TypeId::LookupByName("ns3::OranLmNoop");

        if (useOnnx)
        {
            NS_ABORT_MSG_IF(
                !TypeId::LookupByNameFailSafe("ns3::OranLmLte2LteOnnxHandover", &defaultLmTid),
                "ONNX LM not found. Were ONNX headers/libraries found during configure?");
        }
        else if (useTorch)
        {
            NS_ABORT_MSG_IF(
                !TypeId::LookupByNameFailSafe("ns3::OranLmLte2LteTorchHandover", &defaultLmTid),
                "Torch LM not found. Were Torch headers/libraries found during configure?");
        }
        else if (useDistance)
        {
            defaultLmTid = TypeId::LookupByName("ns3::OranLmLte2LteDistanceHandover");
        }
        else if (useRsrp)
        {
            defaultLmTid = TypeId::LookupByName("ns3::OranLmLte2LteRsrpHandover");
        }
        // Future extension:
        // else if (useLstm)
        // {
        //     NS_ABORT_MSG_IF(
        //         !TypeId::LookupByNameFailSafe("ns3::OranLmLte2LteLstmHandover", &defaultLmTid),
        //         "LSTM LM not found.");
        // }

        ObjectFactory defaultLmFactory;
        defaultLmFactory.SetTypeId(defaultLmTid);
        Ptr<OranLm> defaultLm = defaultLmFactory.Create<OranLm>();

        // --------------------------------------------------
        // Core O-RAN components
        // --------------------------------------------------
        Ptr<OranDataRepository> dataRepository = CreateObject<OranDataRepositorySqlite>();
        Ptr<OranCmm> cmm = CreateObject<OranCmmHandover>();
        Ptr<OranNearRtRic> nearRtRic = CreateObject<OranNearRtRic>();
        Ptr<OranNearRtRicE2Terminator> nearRtRicE2Terminator =
            CreateObject<OranNearRtRicE2Terminator>();

        dataRepository->SetAttribute("DatabaseFile", StringValue(dbFileName));

        defaultLm->SetAttribute("Verbose", BooleanValue(verbose));
        defaultLm->SetAttribute("NearRtRic", PointerValue(nearRtRic));

        // Optional processing delay for ML LM
        if (!processingDelayRv.empty())
        {
            defaultLm->SetAttribute("ProcessingDelayRv", StringValue(processingDelayRv));
        }

        cmm->SetAttribute("NearRtRic", PointerValue(nearRtRic));

        nearRtRicE2Terminator->SetAttribute("NearRtRic", PointerValue(nearRtRic));
        nearRtRicE2Terminator->SetAttribute("DataRepository", PointerValue(dataRepository));
        nearRtRicE2Terminator->SetAttribute(
            "TransmissionDelayRv",
            StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(txDelay) + "]"));

        nearRtRic->SetAttribute("DefaultLogicModule", PointerValue(defaultLm));
        nearRtRic->SetAttribute("E2Terminator", PointerValue(nearRtRicE2Terminator));
        nearRtRic->SetAttribute("DataRepository", PointerValue(dataRepository));
        nearRtRic->SetAttribute("LmQueryInterval", TimeValue(lmQueryInterval));
        nearRtRic->SetAttribute("ConflictMitigationModule", PointerValue(cmm));

        // Optional advanced timing controls, if supported in your branch
        if (useAdvancedRicConfig)
        {
            nearRtRic->SetAttribute("LmQueryMaxWaitTime", TimeValue(maxWaitTime));
            nearRtRic->SetAttribute("LmQueryLateCommandPolicy", StringValue(lateCommandPolicy));
            nearRtRic->SetAttribute("E2NodeInactivityThreshold", TimeValue(Seconds(2)));
        }

        // DB logging to terminal
        if (dbLog)
        {
            nearRtRic->Data()->TraceConnectWithoutContext("QueryRc", MakeCallback(&QueryRcSink));
        }

        Simulator::Schedule(Seconds(1.0), &OranNearRtRic::Start, nearRtRic);

        // Keep terminators alive if needed later
        std::vector<Ptr<OranE2NodeTerminatorLteUe>> ueTerminators;
        std::vector<Ptr<OranE2NodeTerminatorLteEnb>> enbTerminators;

        // --------------------------------------------------
        // UE-side E2 nodes and reporters
        // --------------------------------------------------
        for (uint32_t idx = 0; idx < ueNodes.GetN(); idx++)
        {
            Ptr<Node> ueNode = ueNodes.Get(idx);

            Ptr<OranReporterLocation> locationReporter = CreateObject<OranReporterLocation>();
            Ptr<OranReporterLteUeCellInfo> lteUeCellInfoReporter =
                CreateObject<OranReporterLteUeCellInfo>();
            Ptr<OranReporterAppLoss> appLossReporter = CreateObject<OranReporterAppLoss>();
            Ptr<OranReporterLteUeRsrpRsrq> rsrpRsrqReporter =
                CreateObject<OranReporterLteUeRsrpRsrq>();

            Ptr<OranE2NodeTerminatorLteUe> lteUeTerminator =
                CreateObject<OranE2NodeTerminatorLteUe>();

            // Bind reporters to terminator
            locationReporter->SetAttribute("Terminator", PointerValue(lteUeTerminator));
            lteUeCellInfoReporter->SetAttribute("Terminator", PointerValue(lteUeTerminator));
            appLossReporter->SetAttribute("Terminator", PointerValue(lteUeTerminator));
            rsrpRsrqReporter->SetAttribute("Terminator", PointerValue(lteUeTerminator));

            // App loss traces
            if (idx < remoteApps.GetN() && idx < ueApps.GetN())
            {
                remoteApps.Get(idx)->TraceConnectWithoutContext(
                    "Tx",
                    MakeCallback(&ns3::OranReporterAppLoss::AddTx, appLossReporter));

                ueApps.Get(idx)->TraceConnectWithoutContext(
                    "Rx",
                    MakeCallback(&ns3::OranReporterAppLoss::AddRx, appLossReporter));
            }

            // Attach LTE PHY measurement trace for RSRP/RSRQ
            for (uint32_t netDevIdx = 0; netDevIdx < ueNode->GetNDevices(); netDevIdx++)
            {
                Ptr<LteUeNetDevice> lteUeDevice =
                    ueNode->GetDevice(netDevIdx)->GetObject<LteUeNetDevice>();

                if (lteUeDevice)
                {
                    Ptr<LteUePhy> uePhy = lteUeDevice->GetPhy();
                    if (uePhy)
                    {
                        uePhy->TraceConnectWithoutContext(
                            "ReportUeMeasurements",
                            MakeCallback(&ns3::OranReporterLteUeRsrpRsrq::ReportRsrpRsrq,
                                        rsrpRsrqReporter));
                    }
                }
            }

            lteUeTerminator->SetAttribute("NearRtRic", PointerValue(nearRtRic));
            lteUeTerminator->SetAttribute(
                "RegistrationIntervalRv",
                StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            lteUeTerminator->SetAttribute(
                "SendIntervalRv",
                StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            lteUeTerminator->SetAttribute(
                "TransmissionDelayRv",
                StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(txDelay) + "]"));

            lteUeTerminator->AddReporter(locationReporter);
            lteUeTerminator->AddReporter(lteUeCellInfoReporter);
            lteUeTerminator->AddReporter(appLossReporter);
            lteUeTerminator->AddReporter(rsrpRsrqReporter);

            // Future custom reporters:
            // lteUeTerminator->AddReporter(speedReporter);
            // lteUeTerminator->AddReporter(sinrReporter);
            // lteUeTerminator->AddReporter(delayReporter);

            lteUeTerminator->Attach(ueNode);
            ueTerminators.push_back(lteUeTerminator);

            Simulator::Schedule(Seconds(1.0),
                                &OranE2NodeTerminatorLteUe::Activate,
                                lteUeTerminator);
        }

        // --------------------------------------------------
        // eNB-side E2 nodes and reporters
        // --------------------------------------------------
        for (uint32_t idx = 0; idx < enbNodes.GetN(); idx++)
        {
            Ptr<Node> enbNode = enbNodes.Get(idx);

            Ptr<OranReporterLocation> locationReporter = CreateObject<OranReporterLocation>();
            Ptr<OranE2NodeTerminatorLteEnb> lteEnbTerminator =
                CreateObject<OranE2NodeTerminatorLteEnb>();

            locationReporter->SetAttribute("Terminator", PointerValue(lteEnbTerminator));

            lteEnbTerminator->SetAttribute("NearRtRic", PointerValue(nearRtRic));
            lteEnbTerminator->SetAttribute(
                "RegistrationIntervalRv",
                StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            lteEnbTerminator->SetAttribute(
                "SendIntervalRv",
                StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
            lteEnbTerminator->SetAttribute(
                "TransmissionDelayRv",
                StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(txDelay) + "]"));

            lteEnbTerminator->AddReporter(locationReporter);

            // Future custom eNB reporters:
            // Ptr<OranReporterEnbLoad> enbLoadReporter = CreateObject<OranReporterEnbLoad>();
            // enbLoadReporter->SetAttribute("Terminator", PointerValue(lteEnbTerminator));
            // lteEnbTerminator->AddReporter(enbLoadReporter);

            lteEnbTerminator->Attach(enbNode);
            enbTerminators.push_back(lteEnbTerminator);

            Simulator::Schedule(Seconds(1.0),
                                &OranE2NodeTerminatorLteEnb::Activate,
                                lteEnbTerminator);
        }
    }
    // ========================
    // ORAN END (Unified Block)
    // ========================

    // Trace successful handovers
    Ptr<OutputStreamWrapper> handoverTraceStream = Create<OutputStreamWrapper>("handover.tr", std::ios::out);
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                                  MakeBoundCallback(&HandoverTrace, handoverTraceStream));
    
    // Periodically trace node positions
    Ptr<OutputStreamWrapper> positionTraceStream = Create<OutputStreamWrapper>("positions.tr", std::ios::out);
    Simulator::Schedule(Seconds(1), &PositionTrace, positionTraceStream, ueNodes);
     
    // Trace rsrp, rsrq, and sinr
    Ptr<OutputStreamWrapper> rsrpRsrqSinrTraceStream = Create<OutputStreamWrapper>("rsrp-rsrq-sinr.tr", std::ios::out);
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
    
    Simulator::Schedule(Seconds(0.1), &SaveEnbPositionsToFile, enbNodes);

    /* Enabling Tracing for the simulation scenario */
    lteHelper->EnablePhyTraces();
    lteHelper->EnableMacTraces();
    lteHelper->EnableRlcTraces();
    lteHelper->EnablePdcpTraces();

    Simulator::Stop(simTime);
    Simulator::Run();

    Simulator::Destroy();
    
    return 0;
}