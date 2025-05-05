// Design and implement Parallel DEPTH First Search based on existing algorithms using
// OpenMP. Use a Tree or an undirected graph for BFS.



#include <iostream>
#include <omp.h>
#include <stack>
#include <vector>
#include <chrono>

using namespace std;

class Graph {
public:
    int vertices;
    int edges;
    vector<vector<int>> graph;
    vector<bool> visited;

    // Constructor to take input for the graph
    Graph(int v, int e) : vertices(v), edges(e) {
        graph.resize(vertices);
    }

    void addEdge(int u, int v) {
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    void initialize_visited() {
        visited.assign(vertices, false);
    }

    // Serial DFS
    void dfs(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current = s.top();
            s.pop();
            cout << current << " ";

            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    s.push(neighbor);
                }
            }
        }
    }

    // Parallel DFS
    void parallel_dfs(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current = s.top();
            s.pop();
            cout << current << " ";

            #pragma omp parallel for
            for (int i = 0; i < graph[current].size(); ++i) {
                int neighbor = graph[current][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            s.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    void printGraph() {
        for (int i = 0; i < vertices; i++) {
            cout << i << " -> ";
            for (int j : graph[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    int v, e;
    cout << "Enter the number of vertices: ";
    cin >> v;
    cout << "Enter the number of edges: ";
    cin >> e;

    Graph g(v, e);

    cout << "Enter the edges (u v): \n";
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    cout << "Graph Representation (Adjacency List):\n";
    g.printGraph();

    g.initialize_visited();
    cout << "Serial DFS starting from node 0:\n";
    auto start = chrono::high_resolution_clock::now();
    g.dfs(0);
    auto end = chrono::high_resolution_clock::now();
    cout << endl;
    cout << "Time taken for serial DFS: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    g.initialize_visited();
    cout << "Parallel DFS starting from node 0:\n";
    start = chrono::high_resolution_clock::now();
    g.parallel_dfs(0);
    end = chrono::high_resolution_clock::now();
    cout << endl;
    cout << "Time taken for parallel DFS: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    return 0;
}
