#include "Common.h"

extern "C" {
    long long * init_nnfem_mesh(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int order, int degree, long long *nedges){
        if (mmesh.elements.size()>0){
            printf("WARNING: Internal mesh is being overwritten!\n");
            for(int i = 0; i< mmesh.nelem; i++) delete mmesh.elements[i];
            mmesh.elements.clear();
        }
        if (degree==-1)
            return mmesh.init_BDM1(vertices, num_vertices, element_indices, num_elements, order, nedges);
        return mmesh.init(vertices, num_vertices, element_indices, num_elements, order, degree, nedges);
    }

    // return total number of Gauss points
    int mfem_get_ngauss(){
        return mmesh.ngauss;
    }

    void mfem_get_gauss(double *x, double *y){
        memcpy(x, mmesh.GaussPts.data(), mmesh.ngauss * sizeof(double));
        memcpy(y, mmesh.GaussPts.data() + mmesh.ngauss, mmesh.ngauss * sizeof(double));
    }

    void mfem_get_gauss_weights(double *w){
        int s = 0;
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int k = 0; k < elem->ngauss; k++){
                w[s++] = elem->w[k];
            }
        }
    }

    void mfem_get_area(double *a){
        for(int i = 0; i<mmesh.nelem; i++)
            a[i] = mmesh.elements[i]->area;
    }

    int mfem_get_elem_ndof(){
        return mmesh.elements[0]->ndof;
    }

    int mfem_get_ndof(){
        return mmesh.ndof;
    }

    void mfem_get_connectivity(long long *conn){
        int p = 0;
        for(int i = 0; i<mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for(int k = 0; k < elem->ndof; k++)
                conn[p++] = mmesh.elements[i]->dof[k] + 1;
        }
    }

    void mfem_get_element_to_vertices(long long *elems){
        for (int i = 0; i < mmesh.nelem; i++){
            auto elem = mmesh.elements[i];
            for (int k = 0; k < 3; k++){
                elems[k * mmesh.nelem + i] = elem->node[k] + 1;
            }
        }
    }
}