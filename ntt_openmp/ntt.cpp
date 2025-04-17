#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;
using LL = long long;

const LL P = 998244353; 
const LL G = 3;       

// Fast exponentiation algorithm https://homepages.math.uic.edu/~leon/cs-mcs401-s08/handouts/fastexp.pdf
LL qpow(LL base, LL power) {
    LL result= 1;
    while (power >0) {
        if (power& 1)result = (result * base) % P;
        base = (base*base) % P;
        power >>=1;
    }
    return result;
}

//Find least 2^k such that 2^k > n 
int expand(int n) {
    int m = 1;
    while (m <n) m<<= 1;
    return m;
}

//Parallel NTT op=1 means forward -1 means inverse
void NTT(vector<LL>& A, int n, int op) {
    vector<int> R(n);
    //bit reverse 
    for (int i = 0; i < n; i++) {
        R[i] = (R[i>>1] >> 1) | ((i & 1) ? (n >> 1) : 0);
    }
    
    LL g = G;
    LL gi = qpow(g, P - 2); //Compute the inverse of g which is g^(-1) by fermat little theorem
 
    //Assign the multiplication factor befor butterflies algorithm
    vector<LL> g1_values;
    for (int i = 2; i <= n; i <<= 1) {
        g1_values.push_back(qpow(op == 1 ? g : gi, (P-1) / i));
    }

    LL inv_n = 1; 
    if (op == -1) {
        inv_n = qpow(n, P-2);
    }
       // Butterflies algorithm from bottom to up. if op = 1 it is NNT if op =-1 it is inverse NTT
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            if (i < R[i]) {
                swap(A[i], A[R[i]]);
            }
        }
        int g1_idx = 0;
        for (int i = 2; i <= n; i <<= 1) {
            LL g1 = g1_values[g1_idx++];//obtain the mutiplication factor
            #pragma omp for schedule(dynamic, 32)
            for (int j = 0; j < n; j += i) {
                LL gk = 1;
                for (int k = j; k < j + i / 2; k++) {
                    LL x = A[k];
                    LL y = (gk*A[k + i/2])% P;
                    A[k] = (x+y) %P;
                    A[k + i/2] = (x-y+P) % P;
                    gk = (gk * g1) % P;
                }
            }
 
        }
        
        if (op == -1) {
            #pragma omp for simd schedule(static)
            for (int i = 0; i < n; i++) {
                A[i] = (A[i] * inv_n) % P;
            }
        }
    }
}


// serial version
void serial_NTT(vector<LL>& A, int n, int op) {
    //bit reverse 
    vector<int> R(n);
    for (int i = 0; i < n; i++) {
        R[i] = (R[i>>1] >> 1) | ((i & 1) ? (n >> 1) : 0);
    }
    // rarrange according to above bit reverse
    for (int i=0; i < n; i++) {
        if (i < R[i])swap(A[i], A[R[i]]);
    }
    LL g = G;
    LL gi=qpow(g, P - 2); //Compute the inverse of g which is g^(-1) by fermat little theorem
    // Butterflies algorithm from bottom to up. if op = 1 it is NNT if op =-1 it is inverse NTT
    for (int i = 2; i <= n; i <<= 1) {
        LL g1 = qpow(op ==1?g:gi,(P-1) / i);
        for (int j=0; j<n; j+=i) {
            LL gk=1;
            for (int k = j; k < j + i / 2; k++) {
                LL x=A[k];
                LL y=(gk*A[k +i/2]) % P;
                A[k]=(x +y)%P;
                A[k+i/2]=(x-y + P) % P;
                gk=(gk*g1)% P;
            }
        }
    }
    
    // If it is inverse transform we need to multiply inverse of n
    if (op ==-1) {
        LL inv_n = qpow(n,P-2);
        for (int i =0; i < n; i++) {
            A[i]=(A[i]*inv_n)% P;
        }
    }
}


vector<LL> polynomial_multiply(const vector<LL>& A, const vector<LL>& B) {
    int na = A.size(), nb=B.size();//get the size of na and nb
    int n = na+nb-1;  // obtain the size after multiplication
    int m =expand(n);   //expand to 2^k
    vector<LL> fa(A.begin(),A.end());
    vector<LL> fb(B.begin(),B.end());
    fa.resize(m,0);
    fb.resize(m,0);
    
    //get the NTT transform form
    NTT(fa,m,1);
    NTT(fb,m,1);

    //do the element wise multiplication
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        fa[i] = (fa[i] * fb[i]) % P;
    }
    //inverse ntt transform back to original form
    NTT(fa,m,-1);
    
    fa.resize(n);
    return fa;
}


vector<LL> serial_polynomial_multiply(const vector<LL>& A, const vector<LL>& B) {
    int na = A.size(), nb=B.size();//get the size of na and nb
    int n = na+nb-1;  // obtain the size after multiplication
    int m =expand(n);   //expand to 2^k
    vector<LL> fa(A.begin(),A.end());
    vector<LL> fb(B.begin(),B.end());
    fa.resize(m,0);
    fb.resize(m,0);
    
    //get the NTT transform form
    serial_NTT(fa,m,1);
    serial_NTT(fb,m,1);

    //do the element wise multiplication
    for (int i = 0; i < m; i++) {
        fa[i] = (fa[i] * fb[i]) % P;
    }
    //inverse ntt transform back to original form
    serial_NTT(fa,m,-1);
    
    fa.resize(n);
    return fa;
}

// print polynomial
void print_polynomial(const vector<LL>& poly) {
    for (size_t i = 0; i < poly.size(); i++) {
        cout << poly[i];
        if (i > 0) cout << "x^" << i;
        if (i < poly.size() - 1) cout << " + ";
    }
    cout << endl;
}

vector<LL> read_polynomial() {
    int degree;
    cout << "Input the highest order of polynomial：";
    cin>> degree;
    
    vector<LL> poly(degree + 1);
    cout << "Write down the coefficient from lowest order to highest order, between each coefficient we need space：" << endl;
    for (int i = 0; i <= degree; ++i) {
        cin >> poly[i];
    }
    
    return poly;
}

// Generate random polynomial 
vector<LL> generate_random_polynomial(int degree, unsigned int seed = 0) {
    vector<LL> poly(degree + 1);
    std::mt19937 rng(seed?seed:std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 999);
    for (int i = 0; i <= degree; i++) {
        poly[i]=dist(rng);
    }
    return poly;
}

// Compare if two polynomial are equal? 
bool polynomials_equal(const vector<LL>& A, const vector<LL>& B) {
    for (size_t i = 0; i < A.size(); i++) {
        if (A[i] != B[i]) return false;
    }
    return true;
}

int main() {
    cout << "Polynimial Multiplication program" << endl;
    cout << "----------------------------------------" << endl;
    
    int choice;
    cout << "Choose what you wan to do?" << endl;
    cout << "1. Input two polynomial by hand" << endl;
    cout << "2. Accuracy test（Serial vs Parrallel）" << endl;
    cout << "3. Runtime test with large scale polynomial" << endl;
    cin >> choice;
    
    vector<LL> A, B;
    
    if (choice == 1) {
        cout << "Input first polynomial：" << endl;
        A = read_polynomial();
        
        cout << "Input second polynomial：" << endl;
        B = read_polynomial();
        
        int num_threads;
        cout << "How many threads you need?：(0 means serial)";
        cin >> num_threads;
        
        vector<LL> C;
  
        
        if (num_threads == 0) {
          
            C = serial_polynomial_multiply(A, B);
        }
        else{
            C = polynomial_multiply(A, B);
        }
        cout << "Misssion complete！" << endl;
        cout << "----------------------------------------" << endl;
        
        cout << "The polynomial after multiplication is：" << endl;
        print_polynomial(C);
    } else if (choice == 2) {
        cout << "Accuracy test" << endl;
        cout << "----------------------------------------" << endl;       
        vector<int> test_sizes = {1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
        cout << "We will test with following sclae：";
        for (int size : test_sizes) {
            cout << size << " ";
        }
        cout << endl;

        vector<int> thread_counts = {1, 2, 4, 8, 16};
        cout << "Thread Number：";
        for (int threads : thread_counts) {
            cout << threads << " ";
        }
        cout << endl;
        
        cout << "----------------------------------------" << endl;

        unsigned int seed = time(nullptr);
        cout << "Set random seed: " << seed << endl;
        
        for (int size : test_sizes) {
            cout << "Test scale size: " << size << " order polynomial " << endl;
            vector<LL> poly_A = generate_random_polynomial(size, seed);
            vector<LL> poly_B = generate_random_polynomial(size, seed + 1);
            cout << "Serial output..." << endl;
            vector<LL> serial_result = serial_polynomial_multiply(poly_A, poly_B);

            for (int threads : thread_counts) {
                cout << "  Use " << threads << " threads..." << endl;
                omp_set_num_threads(threads);
                vector<LL> parallel_result =polynomial_multiply(poly_A, poly_B);
                //check serial result with parrallel result
                bool correct= polynomials_equal(serial_result, parallel_result);
                if(correct){
                    cout<< "Result is consistent with serial version."<<endl;
                }else{
                    cout<< "our opnemp version is not correct"<<endl;
                }
            }
            
            cout << "----------------------------------------" << endl;
        }
    } else if (choice == 3) {

        cout << "Run time test" << endl;
        cout << "----------------------------------------" << endl;
        
   
        int num_threads;
        cout << "Input how many threads?(0 means serial version)：";
        cin >> num_threads;
        
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        vector<int> test_sizes = {1<<8, 1<<12, 1<<16, 1<<20, 1<<24, 1<<26};
        cout << "We will test following different order scale：";
        for (int size : test_sizes) {
            cout << size << " ";
        }
        cout << endl;
        
        for (int size : test_sizes) {
            cout << "Test size: " << size << "order polynomial" << endl;
            vector<LL> large_A(size);
            vector<LL> large_B(size);
            
            for (int i = 0; i < size; i++) {
                large_A[i] = rand() % 1000;
                large_B[i] = rand() % 1000;
            }
            
            if (num_threads == 0) {

                cout << "Start serial computation..." << endl;
                double serial_start = omp_get_wtime();
                vector<LL> result = serial_polynomial_multiply(large_A, large_B);
                double serial_end = omp_get_wtime();
                double serial_time = (serial_end - serial_start)*1000 ; 
                cout << "Serial Mission comoplete！execution time：" << fixed << setprecision(2) << serial_time << " milliseconds";
                if (serial_time > 1000) {
                    cout << " (" << serial_time / 1000.0 << " seconds)";
                }
                cout << endl;
                
              
            } else {
                cout << "Start computation..." << endl;
                double parallel_start = omp_get_wtime();
                vector<LL> result = polynomial_multiply(large_A, large_B);
                double parallel_end = omp_get_wtime();
                double parallel_time = (parallel_end-parallel_start) * 1000; 
                
                cout << "Parrallel Mission comoplete！execution time：" << fixed << setprecision(2) << parallel_time << " milliseconds";
                if (parallel_time > 1000) {
                    cout << " (" << parallel_time / 1000.0 << " seconds)";
                }
                cout << endl;
            }
            
            cout << "----------------------------------------" << endl;
        }
    }
    
    return 0;
}