#include <eigen3/Eigen>
#include <vector>
#include <set>

class ZeroOutRow{
    private:
        std::vector<int64> ii;
        std::vector<int64> jj;
        std::vector<double> vv;
        const int64 *indices;
        const double *val;
        const int64 *bd;
        int N;
    public:
        MatrixTransform(){

        }
        void forward();
}

void MatrixTransform::forward(){
    std::set<int64> bdset;
    for (int i = 0; i < nbd; i++){
        bdset.insert(bd[i]);
    }
    for (int i = 0; i < N; i++){
        
    }
}

