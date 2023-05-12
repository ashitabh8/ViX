// Discrete DARE infinite horizon 
#undef abs
#undef max
#undef min
#undef round


#include <random>
// #include <iostream>
#include <array>

// #include "../../include/ios.hpp"
#include "fixed.hpp"
#include "math.hpp"

// #define T float

#define T fpm::fixed<std::int32_t, std::int64_t, 11>

//#define T double

void transpose(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &AT)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            AT[i][j] = A[j][i];
        }
    }
}

T determinant(std::array<std::array<T, 3>, 3> &A)
{
    T det = 0;
    for(int i = 0; i < 3; i++)
    {
        det += (A[0][i] * (A[1][(i+1)%3] * A[2][(i+2)%3] - A[1][(i+2)%3] * A[2][(i+1)%3]));
    }
    return det;
}

void inverse(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &A_inv)
{
    T det = determinant(A);
    std::array<std::array<T, 3>, 3> A_adj;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            A_adj[i][j] = (A[(i+1)%3][(j+1)%3] * A[(i+2)%3][(j+2)%3] - A[(i+1)%3][(j+2)%3] * A[(i+2)%3][(j+1)%3]);
        }
    }
    // std::cout << "det: " << det << std::endl;
    std::array<std::array<T, 3>, 3> AT;
    transpose(A_adj, AT);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            A_inv[i][j] = AT[i][j] / det;
        }
    }
}

// Add matrices

void add(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B, std::array<std::array<T, 3>, 3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Multiply matrices

void multiply(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B, std::array<std::array<T, 3>, 3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = 0;
            for(int k = 0; k < 3; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Subtract matrices

void subtract(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B, std::array<std::array<T, 3>, 3> &C)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Print matrix
void print(std::array<std::array<T, 3>, 3> &A)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            Serial.print((float) A[i][j]);
            Serial.print(" ");
        }
        Serial.println();
    }
}


void print(std::array<T, 3> &A)
{
    for(int i = 0; i < 3; i++)
    {
        Serial.print((float) A[i]);
        Serial.print(" ");
    }
    Serial.println();
}

// copy elements of matrix A to matrix B
void copy(std::array<std::array<T, 3>, 3> &A, std::array<std::array<T, 3>, 3> &B)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            B[i][j] = A[i][j];
        }
    }
}

// Matrix vector multiplication
void multiply_mat_vector(std::array<std::array<T, 3>, 3> &A, std::array<T, 3> &x, std::array<T, 3> &y)
{
    for(int i = 0; i < 3; i++)
    {
        y[i] = 0;
        for(int j = 0; j < 3; j++)
        {
            y[i] += A[i][j] * x[j];
        }
    }
}


// Infinite horizon discrete DARE
void DARE(std::array<std::array<T,3>,3> &A, std::array<std::array<T,3>,3> &B, std::array<std::array<T,3>,3> &Q, 
        std::array<std::array<T,3>,3> &R, std::array<std::array<T,3>,3> &Pk, 
        std::array<std::array<T,3>,3> &Pk_1)
{
    std::array<std::array<T,3>,3> AT, BT, AT_Pk, AT_Pk_A, Q_add_At_P_A,  AT_Pk_B,BT_Pk, BT_Pk_B, R_add_Bt_P_B, R_add_Bt_P_B_inv, BT_Pk_A,
     R_add_Bt_P_B_inv_mul_BT_Pk_A, AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A,
     At_Pk_A_sub_AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A;

    transpose(A, AT);
    transpose(B, BT);
    multiply(AT, Pk, AT_Pk);
    multiply(AT_Pk, A, AT_Pk_A);
    add(Q, AT_Pk_A, Q_add_At_P_A);
    multiply(AT_Pk, B, AT_Pk_B);
    multiply(BT, Pk, BT_Pk);
    multiply(BT_Pk, B, BT_Pk_B);
    add(R, BT_Pk_B, R_add_Bt_P_B);
    inverse(R_add_Bt_P_B, R_add_Bt_P_B_inv);
    multiply(BT_Pk, A, BT_Pk_A);
    multiply(R_add_Bt_P_B_inv, BT_Pk_A, R_add_Bt_P_B_inv_mul_BT_Pk_A);
    multiply(AT_Pk_B, R_add_Bt_P_B_inv_mul_BT_Pk_A, AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A);
    subtract(AT_Pk_A, AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A, At_Pk_A_sub_AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A);
    add(Q, At_Pk_A_sub_AT_Pk_B_mul_R_add_Bt_P_B_inv_mul_BT_Pk_A, Pk_1);

}

// multiply scalar and matrix

void scalar_mat_mul(T scalar, std::array<std::array<T,3>,3> &A, std::array<std::array<T,3>,3> &B)
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            B[i][j] = scalar * A[i][j];
        }
    }
}


void LQR(std::array<std::array<T,3>,3> &A, std::array<std::array<T,3>,3> &B, std::array<std::array<T,3>,3> &Q, 
        std::array<std::array<T,3>,3> &R, std::array<std::array<T,3>,3> &Pk, 
        std::array<std::array<T,3>,3> &Pk_1, std::array<std::array<T,3>,3> &N, std::array<T,3> &X, std::array<std::array<T,3>,3> &K)
{
    copy(Q, Pk);
    for(int i = 0; i < 1; i++)
    {
        DARE(A, B, Q, R, Pk, Pk_1);
        copy(Pk_1, Pk);
    }
    copy(Pk, Pk_1);
    // DARE(A, B, Q, R, Pk, Pk_1);
    std::array<std::array<T,3>,3> Nt, Bt, Bt_Pk_1, Bt_Pk_1_B, R_add_Bt_Pk_1_B, R_add_Bt_Pk_1_B_inv, Bt_Pk_1_A, N_add_Bt_Pk_1_A, F;
    transpose(B, Bt);
    transpose(N, Nt);
    multiply(Bt, Pk_1, Bt_Pk_1);
    multiply(Bt_Pk_1, B, Bt_Pk_1_B);
    add(R, Bt_Pk_1_B, R_add_Bt_Pk_1_B);
    inverse(R_add_Bt_Pk_1_B, R_add_Bt_Pk_1_B_inv);
    multiply(Bt_Pk_1, A, Bt_Pk_1_A);
    add(Nt, Bt_Pk_1_A, N_add_Bt_Pk_1_A);
    multiply(R_add_Bt_Pk_1_B_inv, N_add_Bt_Pk_1_A, F);
    scalar_mat_mul(-1, F, K);
}



void setup()
{
    Serial.begin(9600);
    while (!Serial) {
      ; // wait for serial port to connect. Needed for native USB port only
    }
  // put your setup code here, to run once:

    Serial.println("LQR Testing");

    std::array<std::array<T,3>,3> A{{{1,0.1,0},{0,1,0.1},{0,0,0}}};

    std::array<std::array<T,3>,3> A_approx{{{1,0.1,0},{0,1,0.1},{0,0,0.001}}};

    std::array<std::array<T,3>,3> B{{{1,0,0},{0,1,0},{0,0,1}}};

    std::array<std::array<T,3>,3> Q{{{1,0,0},{0,0.1,0},{0,0,0.1}}};

    std::array<std::array<T,3>,3> R{{{100,0,0},{0,100,0},{0,0,1}}}; // changed from the original.

    std::array<std::array<T,3>,3> N{{{0,0,0},{0,0,0},{0,0,0}}};

    std::array<T,3> x_init{50, 0, 0.5};
    std::array<T,3> u;

    std::array<T,3> noise{0.01, 0.01, 0.01};

    std::array<std::array<T,3>,3> Pk, Pk_1;
    // Init Pk = Q
    
    // copy(Q, Pk);
    // for(int i = 0; i < 100; i++)
    // {   
    //     // std::cout << "*** Iteration " << i << " ***" << std::endl;
    //     DARE(A, B, Q, R, Pk, Pk_1);
        
    //     copy(Pk_1, Pk);
    // }
    // print(Pk_1);

    std::array<std::array<T,3>,3> K;

    unsigned long int start = micros();
    LQR(A, B, Q, R, Pk, Pk_1, N, x_init, K);
    unsigned long int end = micros();
    Serial.print("Time taken for LQR (float): ");
    Serial.println((float) end - start);
    Serial.println("K");
    print(K);
    Serial.println();
    multiply_mat_vector(K, x_init, u);
    print(u);

}

void loop()
{

}
