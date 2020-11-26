#include "Common.h"

extern "C" {
    long long * init_nnfem_mesh3(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int order, int degree, long long *nedges){
        if (mmesh3.elements.size()>0){
            printf("WARNING: Internal mesh is being overwritten!\n");
            for(int i = 0; i< mmesh3.nelem; i++) delete mmesh3.elements[i];
            mmesh3.elements.clear();
        }
        return mmesh3.init(vertices, num_vertices, element_indices, num_elements, order, degree, nedges);
    }

    // return total number of Gauss points
    int mfem_get_ngauss3s(){
        return mmesh3.ngauss;
    }

    void mfem_get_gauss3(double *x, double *y, double *z){
        memcpy(x, mmesh3.GaussPts.data(), mmesh3.ngauss * sizeof(double));
        memcpy(y, mmesh3.GaussPts.data() + mmesh3.ngauss, mmesh3.ngauss * sizeof(double));
        memcpy(z, mmesh3.GaussPts.data() + 2*mmesh3.ngauss, mmesh3.ngauss * sizeof(double));
    }

    void mfem_get_gauss_weights3(double *w){
        int s = 0;
        for (int i = 0; i < mmesh3.nelem; i++){
            auto elem = mmesh3.elements[i];
            for (int k = 0; k < elem->ngauss; k++){
                w[s++] = elem->w[k];
            }
        }
    }

    void mfem_get_volume3(double *a){
        for(int i = 0; i<mmesh3.nelem; i++)
            a[i] = mmesh3.elements[i]->volume;
    }

    int mfem_get_elem_ndof3(){
        return mmesh3.elements[0]->ndof;
    }

    int mfem_get_ndof3(){
        return mmesh3.ndof;
    }

    void mfem_get_connectivity3(long long *conn){
        int p = 0;
        for(int i = 0; i<mmesh3.nelem; i++){
            auto elem = mmesh3.elements[i];
            for(int k = 0; k < elem->ndof; k++)
                conn[p++] = mmesh3.elements[i]->dof[k] + 1;
        }
    }

    void mfem_get_element_to_vertices3(long long *elems){
        for (int i = 0; i < mmesh3.nelem; i++){
            auto elem = mmesh3.elements[i];
            for (int k = 0; k < 4; k++){
                elems[k * mmesh3.nelem + i] = elem->node[k] + 1;
            }
        }
    }
}


void test_APIs(){
    double vertices[] = {0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 
                        1.0,1.0,1.0,
                        0.0,1.0,1.0};
    int element_indices[] = {
        0,1,2,3
    };
    long long nedges;
    int degree = 2;
    int quad_order = 2;
    auto *edges = init_nnfem_mesh3(vertices, 4, element_indices, 1, quad_order, degree, &nedges);
    printf("Edges = ");
    int k = 0;
    for(int i = 0; i < nedges; i ++){
        printf("%d, %d\n", edges[k]-1, edges[k+nedges]-1);
        k ++;
    }
    printf("\n");

    int ngauss = mfem_get_ngauss3s();
    double v;
    printf("ngauss = %d\n", ngauss);
    double x[ngauss], y[ngauss], z[ngauss], w[ngauss];
    mfem_get_gauss3(x, y, z);
    mfem_get_gauss_weights3(w);
    printf("Guass = ");
    for(int i = 0; i < ngauss; i ++){
        printf("%f %f %f %f\n", x[i], y[i], z[i], w[i]);
    }
    printf("\n");

    mfem_get_volume3(&v);
    printf("volume = %f\n", v);

    int n = mfem_get_elem_ndof3();
    printf("element_ndof = %d\n", n);

    n = mfem_get_ndof3();
    printf("ndof = %d\n", n);

    long long conn[10];
    mfem_get_connectivity3(conn);
    printf("Connectivity = ");
    for(int i = 0; i < 10; i ++){
        printf("%d ", conn[i]);
    }
    printf("\n");

    long long elems[4];
    mfem_get_element_to_vertices3(elems);
    printf("Elements = ");
    for(int i = 0; i < 4; i ++){
        printf("%d ", elems[i]);
    }
    printf("\n");
}

int main(){
    test_APIs();
    return 1;
}