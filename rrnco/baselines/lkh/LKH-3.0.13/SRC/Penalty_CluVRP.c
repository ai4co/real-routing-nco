#include "LKH.h"
#include "Segment.h"

GainType Penalty_CluVRP()
{
    static Node *StartRoute = 0;
    Node *N, *CurrentRoute;
    GainType DemandSum, DistanceSum, P = 0;
    int *ColorUsed, i;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > DimensionSaved)
        StartRoute -= DimensionSaved;
    N = StartRoute;
    ColorUsed = (int *) malloc((GVRPSets + 1) * sizeof(int));
    do {
        int Size = -1;
        int CurrentColor = 0;
        CurrentRoute = N;
        DemandSum = 0;
        memset(ColorUsed, 0, (GVRPSets + 1) * sizeof(int));
        do {
            DemandSum += N->Demand;
            Size++;
            ColorUsed[N->Color]++;
            if (ProblemType == SoftCluVRP)
                continue;
            if (CurrentColor && N->Color && N->Color != CurrentColor) {
                P += ColorCount[CurrentColor] -
                     ColorUsed[CurrentColor];
                if (P > CurrentPenalty ||
                    (P == CurrentPenalty && CurrentGain <= 0)) {
                    StartRoute = CurrentRoute;
                    free(ColorUsed);
                    return CurrentPenalty + (CurrentGain > 0);
                }
                CurrentColor = N->Color;
            }
        } while ((N = SUCC(N))->DepotId == 0);
        for (i = 1; i <= GVRPSets; i++)
            if (ColorUsed[i])
                P += ColorCount[i] - ColorUsed[i];
        if (MTSPMinSize >= 1 && Size < MTSPMinSize)
            P += MTSPMinSize - Size;
        if (Size > MTSPMaxSize)
            P += Size - MTSPMaxSize;
        if (DemandSum > Capacity)
            P += DemandSum - Capacity;
        if (P > CurrentPenalty ||
            (P == CurrentPenalty && CurrentGain <= 0)) {
            StartRoute = CurrentRoute;
            free(ColorUsed);
            return CurrentPenalty + (CurrentGain > 0);
        }
        if (DistanceLimit != DBL_MAX) {
            DistanceSum = 0;
            N = CurrentRoute;
            do {
                DistanceSum += (C(N, SUCC(N)) - N->Pi - SUCC(N)->Pi) /
                    Precision;
                if (!N->DepotId)
                    DistanceSum += N->ServiceTime;
            } while ((N = SUCC(N))->DepotId == 0);
            if (DistanceSum > DistanceLimit &&
                ((P += DistanceSum - DistanceLimit) > CurrentPenalty ||
                 (P == CurrentPenalty && CurrentGain <= 0))) {
                StartRoute = CurrentRoute;
                free(ColorUsed);
                return CurrentPenalty + (CurrentGain > 0);
            }
        }
    } while (N != StartRoute);
    free(ColorUsed);
    return P;
}
