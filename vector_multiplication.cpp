//Write a program to implement Parallel matrix vector multiplication using OpenMp.
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

void sequentialMatrixVector(const vector<vector<int>>& mat, const vector<int>& vec, vector<int>& result) {
    int n = mat.size();
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int j = 0; j < vec.size(); j++)
            result[i] += mat[i][j] * vec[j];
    }
}

void parallelMatrixVector(const vector<vector<int>>& mat, const vector<int>& vec, vector<int>& result) {
    int n = mat.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int j = 0; j < vec.size(); j++)
            result[i] += mat[i][j] * vec[j];
    }
}

void printVector(const vector<int>& v) {
    for (auto x : v)
        cout << x << " ";
    cout << endl;
}

int main() {
    int n, m;
    cout << "Enter number of rows: ";
    cin >> n;
    cout << "Enter number of columns: ";
    cin >> m;
    
    vector<vector<int>> mat(n, vector<int>(m));
    vector<int> vec(m);
    vector<int> result_seq(n), result_par(n);
    
    cout << "Enter matrix elements:\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            cout << "Element [" << i << "][" << j << "]: ";
            cin >> mat[i][j];
        }
        
    cout << "Enter vector elements:\n";
    for (int i = 0; i < m; ++i) {
        cout << "Vector Element " << i+1 << ": ";
        cin >> vec[i];
    }
    
    auto startSeq = chrono::high_resolution_clock::now();
    sequentialMatrixVector(mat, vec, result_seq);
    auto endSeq = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> durationSeq = endSeq - startSeq;
    
    auto startPar = chrono::high_resolution_clock::now();
    parallelMatrixVector(mat, vec, result_par);
    auto endPar = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> durationPar = endPar - startPar;
    
    cout << "\n----- Sequential Matrix-Vector Multiplication Output -----\n";
    cout << "Result Vector: ";
    printVector(result_seq);
    cout << "Time Taken: " << durationSeq.count() << " milliseconds\n";

    cout << "\n----- Parallel Matrix-Vector Multiplication Output -----\n";
    cout << "Result Vector: ";
    printVector(result_par);
    cout << "Time Taken: " << durationPar.count() << " milliseconds\n";
    
    return 0;
}
