#include <vector>
#include <map>

namespace MFEM{
    class ImposeDirichlet{
    public:
        std::vector<int64> ii;
        std::vector<int64> jj;
        std::vector<double> vv;
        std::vector<double> rhs; 
        void Copy(int64 *oindices, double *ov, double *orhs);
        void ImposeDirichlet_forward(const int64 *indices, const double *v_ipt, const int64 *bd, 
            const double *rhs_ipt, const double *bdval, int N, int bdN, int sN);
        void ImposeDirichlet_backward(
            double *grad_vv, double *grad_rhs, 
            const double *grad_vv_ipt, const double *grad_rhs_ipt, 
            const int64 *indices, const double *v_ipt, const int64 *bd, 
            const double *rhs_ipt, const double *bdval, int N, int bdN, int sN);
        void ImposeDirichlet_backward(
            double *grad_vv_ipt, double *grad_rhs_ipt, double *grad_bdval,
            const double *grad_vv, const double *grad_rhs, 
            const int64 *indices, const double *v_ipt, const int64 *bd, 
            const double *rhs_ipt, const double *bdval, int N, int bdN, int sN);
        int N;
    };

    void ImposeDirichlet::ImposeDirichlet_forward(const int64 *indices, const double *v_ipt, const int64 *bd, 
        const double *rhs_ipt, const double *bdval, int N, int bdN, int sN){
        ImposeDirichlet::N = N;
        std::map<int64, double> bdMap;
        for (int i = 0; i < N; i++) rhs.push_back(rhs_ipt[i]);
        for(int i = 0; i < bdN; i++) bdMap[bd[i]-1] = bdval[i];
        for(int k = 0; k < sN; k++){
            int64 i = indices[2*k], j = indices[2*k+1];
            if (bdMap.count(i)==0 && bdMap.count(j)==0){
                ii.push_back(i);
                jj.push_back(j);
                vv.push_back(v_ipt[k]);
            }

            if (bdMap.count(i)==0 && bdMap.count(j)>0){
                rhs[i] = rhs[i] - v_ipt[k] * bdMap[j];
            }
        }
        for(auto &it: bdMap){
            ii.push_back(it.first);
            jj.push_back(it.first);
            vv.push_back(1.0);
            rhs[it.first] = it.second;
        }
    }

    void ImposeDirichlet::Copy(int64 *oindices, double *ov, double *orhs){
        for(int i = 0; i < N; i++) orhs[i] = rhs[i];
        for(int i = 0; i < ii.size(); i++){
            oindices[2*i] = ii[i];
            oindices[2*i+1] = jj[i];
            ov[i] = vv[i];
        }
    }


    void ImposeDirichlet::ImposeDirichlet_backward(
            double *grad_vv_ipt, double *grad_rhs_ipt, double *grad_bdval,
            const double *grad_vv, const double *grad_rhs, 
            const int64 *indices, const double *v_ipt, const int64 *bd, 
            const double *rhs_ipt, const double *bdval, int N, int bdN, int sN){
        std::map<int64, double> bdMap;
        std::map<int64, int64> bdMapIdx;
        for(int i = 0; i < bdN; i++) bdMap[bd[i]-1] = bdval[i];
        for(int i = 0; i < bdN; i++) bdMapIdx[bd[i]-1] = i;
        int z = 0;
        for(int k = 0; k < sN; k++){
            int64 i = indices[2*k], j = indices[2*k+1];
            if (bdMap.count(i)==0 && bdMap.count(j)==0){
                grad_vv_ipt[k] = grad_vv[z++];
            }

            if (bdMap.count(i)==0 && bdMap.count(j)>0){
                // rhs[i] = rhs[i] - v_ipt[k] * bdMap[j];
                grad_vv_ipt[k] -= bdMap[j] * grad_rhs[i];
                grad_bdval[bdMapIdx[j]] -= v_ipt[k] * grad_rhs[i];
            }
        }
        for (int i = 0; i<N; i++){
            if (bdMap.count(i)==0){
                grad_rhs_ipt[i] = grad_rhs[i];
            }
        }
        for(auto &it: bdMapIdx){
            grad_bdval[it.second] += grad_rhs[it.first];
        }  
    }
}


// J: indof x outdof 
extern "C" void pcl_ImposeDirichlet(
        double *J, 
        const int64 *indices, const int64 *bd, int bdN, int sN){
        std::set<int64> bdMap;
        for(int i = 0; i < bdN; i++) bdMap.insert(bd[i]-1);
        int s = 0;
        for(int k = 0; k < sN; k++){
            int64 i = indices[k] - 1, j = indices[k+sN] - 1;
            if (bdMap.count(i)==0 && bdMap.count(j)==0){
                int idx = k + s * sN;
                s += 1;
                J[idx] = 1.0;
            }
        }
}