#include "LKH.h"
#include "Segment.h"

GainType Penalty_CBnTSP()
{
    static Node *StartRoute = 0;
    Node *N, *N1, *N2, *CurrentRoute;
    GainType P = 0;
    int Forward, Cost, MaxCost = INT_MIN;

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
        do {
            if (N->Color != 0 && N->Color != CurrentRoute->DepotId)
                P += 100000000L;
            N2 = Forward ? SUCC(N) : PREDD(N);
            Cost = (C(N, N2) - N->Pi - N2->Pi) / Precision;
            if (Cost > MaxCost)
                MaxCost = Cost;
            if (P + MaxCost > CurrentPenalty ||
                (P + MaxCost == CurrentPenalty && CurrentGain <= 0)) {
                StartRoute = CurrentRoute;
                return CurrentPenalty + (CurrentGain > 0);
            }
        } while ((N = N2)->DepotId == 0);
    } while (N != StartRoute);
    return P + MaxCost;
}
