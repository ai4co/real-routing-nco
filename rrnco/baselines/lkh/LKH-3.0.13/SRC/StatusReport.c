#include "LKH.h"

void StatusReport(GainType Cost, double EntryTime, char *Suffix)
{
    if (Penalty) {
        printff("Cost = " GainFormat "_" GainFormat, CurrentPenalty, Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0) {
            if (OptimizePenalty)
                printff(", Gap = %0.4f%%",
                        (ProblemType == MSCTSP ? -1 : 1) *
                        100.0 * (CurrentPenalty - Optimum) / Optimum);
            else
                printff(", Gap = %0.4f%%",
                        100.0 * (Cost - Optimum) / Optimum);
        }
        printff(", Time = %0.2f sec. %s",
                fabs(GetTime() - EntryTime), Suffix);
    } else {
        printff("Cost = " GainFormat, Cost);
        if (Optimum != MINUS_INFINITY && Optimum != 0)
            printff(", Gap = %0.4f%%",
                    100.0 * (Cost - Optimum) / Optimum);
        printff(", Time = %0.2f sec.%s%s",
                fabs(GetTime() - EntryTime), Suffix,
                Cost < Optimum ? "<" : Cost == Optimum ? " =" : "");
    }
    printff("\n");
}
