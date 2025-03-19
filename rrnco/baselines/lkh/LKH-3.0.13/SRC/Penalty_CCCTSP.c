#include "LKH.h"
#include "Segment.h"

GainType Penalty_CCCTSP()
{
    static Node *StartRoute = 0;
    Node *N, *N1, *N2, *NextN, *CurrentRoute;
    GainType DemandSum, CostSum, P = 0;
    int Forward;

    N1 = Depot;
    while ((N1 = SUCC(N1))->DepotId == 0);
    N2 = Depot;
    while ((N2 = PREDD(N2))->DepotId == 0);
    Forward = N1 != N2 ? N1->DepotId < N2->DepotId : !Reversed;

    if (!StartRoute)
        StartRoute = Depot;
    N = StartRoute;
    do {
        CurrentRoute = N;
        DemandSum = CostSum = 0;
        do {
            DemandSum += N->Demand;
            if (DemandSum > Capacity)
                P += 10000000000L * (DemandSum - Capacity);
            if (N->DepotId == 0 && !N->ColorAllowed[CurrentRoute->DepotId])
                P += 10000000000L;
            NextN = Forward ? SUCC(N) : PREDD(N);
            if (NextN->DepotId == 0) {
                CostSum += (C(N, NextN) - N->Pi - NextN->Pi) / Precision;
                P += CostSum;
            }
            if (P > CurrentPenalty ||
                (P == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
        } while ((N = NextN)->DepotId == 0);
    } while (N != StartRoute);
    return P;
}
