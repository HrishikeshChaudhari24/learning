// Write a program to implement Parallel matrix matrix multiplication using OpenMp.
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

void sequentialMatrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int n = A.size();
    int m = B[0].size();
    int p = A[0].size();
    
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

void parallelMatrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int n = A.size();
    int m = B[0].size();
    int p = A[0].size();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

void printMatrix(const vector<vector<int>>& mat) {
    for (auto row : mat) {
        for (auto val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main() {
    int n, m, p;
    cout << "Enter dimensions (A: n x p, B: p x m)\n";
    cout << "Enter n: ";
    cin >> n;
    cout << "Enter p: ";
    cin >> p;
    cout << "Enter m: ";
    cin >> m;
    
    vector<vector<int>> A(n, vector<int>(p));
    vector<vector<int>> B(p, vector<int>(m));
    vector<vector<int>> C_seq(n, vector<int>(m, 0));
    vector<vector<int>> C_par(n, vector<int>(m, 0));
    
    cout << "Enter matrix A elements:\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j) {
            cout << "A[" << i << "][" << j << "]: ";
            cin >> A[i][j];
        }
        
    cout << "Enter matrix B elements:\n";
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < m; ++j) {
            cout << "B[" << i << "][" << j << "]: ";
            cin >> B[i][j];
        }
    
    auto startSeq = chrono::high_resolution_clock::now();
    sequentialMatrixMultiply(A, B, C_seq);
    auto endSeq = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> durationSeq = endSeq - startSeq;
    
    auto startPar = chrono::high_resolution_clock::now();
    parallelMatrixMultiply(A, B, C_par);
    auto endPar = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> durationPar = endPar - startPar;
    
    cout << "\n----- Sequential Matrix-Matrix Multiplication Output -----\n";
    printMatrix(C_seq);
    cout << "Time Taken: " << durationSeq.count() << " milliseconds\n";

    cout << "\n----- Parallel Matrix-Matrix Multiplication Output -----\n";
    printMatrix(C_par);
    cout << "Time Taken: " << durationPar.count() << " milliseconds\n";
    
    return 0;
}
