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


    // void ImposeDirichlet::ImposeDirichlet_backward(
    //         double *grad_vv, double *grad_rhs, 
    //         const double *grad_vv_ipt, const double *grad_rhs_ipt, 
    //         const int64 *indices, const double *v_ipt, const int64 *bd, 
    //         const double *rhs_ipt, const double *bdval, int N, int bdN, int sN){
    //     std::map<int64, double> bdMap;
    //     for (int i = 0; i < N; i++) rhs.push_back(rhs_ipt);
    //     for(int i = 0; i < bdN; i++) bdMap[bd[i]-1] = bdval[i];
    //     for(int k = 0; k < sN; k++){
    //         int64 i = indices[2*k], j = indices[2*k+1];
    //         if (bdMap.count(i)==0 && bdMap.count(j)==0){
    //             ii.push_back(i);
    //             jj.push_back(j);
    //             vv.psuh_back(v_ipt[k]);
    //         }

    //         if (bdMap.count(i)==0 && bdMap.count(j)>0){
    //             rhs[i] = rhs[i] - v_ipt[k] * bdMap[j];
    //         }
    //     }
    //     for(auto &it: bdMap){
    //         ii.push_back(it->first);
    //         jj.push_back(it->first);
    //         vv.push_back(1.0);
    //         rhs[i] = it->second;
    //     }  
    // }
}
