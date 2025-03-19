#include "LKH.h"
#include "Segment.h"

GainType Penalty_PTSP1();
GainType Penalty_PTSP2();
GainType Penalty_PTSP3();

GainType Penalty_PTSP() {
    return Penalty_PTSP3();
}

GainType Penalty_PTSP1()
{
    int *t;
    double **Q;
    Node *N;
    int i, r, s, n = Dimension;
    double P = 0, p = Probability / 100.0;

    t = (int *) malloc(n * sizeof(int));
    N = FirstNode;
    i = 0;
    do
        t[i++] = N->Id - 1;
    while ((N = SUCC(N)) != FirstNode);
    Q = (double **) malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
        Q[i] = (double *) malloc(n * sizeof(double));
    for (i = 0; i < n; i++) {
        Q[t[i]][t[i]] = 0.0;
        Q[t[i]][t[(i + 1) % n]] = 1.0;
        for (s = 2; s < n; s++)
            Q[t[i]][t[(i + s) % n]] =
                Q[t[i]][t[(i + s - 1) % n]] * (1 - p);
    }
    for (i = 0; i < n; i++)
        for (r = 0; r < n; r++)
            if (i != r)
                P += Distance(&NodeSet[t[i] + 1], &NodeSet[t[r] + 1]) *
                     p * p * Q[t[i]][t[r]];
    for (i = 0; i < n; i++)
        free(Q[i]);
    free(Q);
    free(t);
    return 100.0 * P;
}

GainType Penalty_PTSP2()
{
    Node *N, **T;
    int i, j, k, n = Dimension;
    double P = 0, Sum, Product, p = Probability / 100.0;

    T = (Node **) malloc(n * sizeof(Node *));
    N = FirstNode;
    i = 0;
    do
        T[++i] = N;
    while ((N = SUCC(N)) != FirstNode);
    assert(i == Dimension);

    for (i = 1; i < n; i++) {
        Sum = 0;
        for (j = i + 1; j <= n; j++) {
            Product = 1;
            for (k = i + 1; k < j; k++)
                Product *= 1 - p;
            Sum += Distance(T[i], T[j]) * p * p * Product;
        }
        P += Sum;
    }
    for (i = 1; i <= n; i++) {
        Sum = 0;
        for (j = 1; j < i; j++) {
            Product = 1;
            for (k = 1; k < j; k++)
                Product *= 1 - p;
            Sum += Distance(T[i], T[j]) * p * p * Product;
        }
        Product = 1;
        for (k = i + 1; k <= n; k++)
            Product *= 1 - p;
        P += Sum * Product;
    }
    free(T);
    return 100 * P;
}

GainType Penalty_PTSP3()
{
    Node *N, **T;
    int i = 0, r, n = Dimension;
    double P = 0, p = Probability / 100.0;
    const double pp100 = p * p * 100;

    T = (Node **) malloc(n * sizeof(Node *));
    N = FirstNode;
    do
        T[i++] = N;
    while ((N = SUCC(N)) != FirstNode);
    for (r = 1; r < n; r++) {
        GainType L = 0;
        for (i = 0; i < n; i++) {
            L += Distance(T[i], T[(i + r) % n]);
            assert(i != (i + r) % n);
        }
        P += pow(1 - p, r - 1) * L;
        if ((GainType) (pp100 * P) > CurrentPenalty)
            break;
    }
    free(T);
    return pp100 * P;
}
