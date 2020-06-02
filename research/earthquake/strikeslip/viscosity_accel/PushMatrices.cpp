#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

using namespace tensorflow;
#include "Common.h"

void push_matrix(int64 *flag, const int64 *ii1, const int64 *jj1, const double *vv1, int n1, 
                const int64 *ii2, const int64 *jj2, int n2, int d){

    if (iiK.size()==0) count = 0;

    std::vector<T> triplets;

    for(int i=0;i<n1;i++){
      // std::cout << ii1[i]-1 << ' ' << jj1[i]-1 << ' ' << vv1[i] << std::endl;
        triplets.push_back(T(ii1[i]-1, jj1[i]-1, vv1[i]));
    }
    SpMat A(d, d); 

    A.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << Eigen::MatrixXd(A) << std::endl;
    // std::cout << A << std::endl;
    solver.analyzePattern(A);
    solver.factorize(A);
    // printf("A Factorized!\n");

    SpMat B = A.transpose();
    // std::cout << A << std::endl;
    solvert.analyzePattern(B);
    solvert.factorize(B);
    // printf("B Factorized!\n");

    iiK.resize(n2); 
    jjK.resize(n2);
    for(int i=0;i<n2;i++){
      iiK[i] = ii2[i]-1;
      jjK[i] = jj2[i]-1;
    }

    // printf("size of K value vectors: %d\n", n2);

    *flag = (count++);

    


}


REGISTER_OP("PushMatrices")

.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("d : int64")
.Output("flag : int64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle ii2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &ii2_shape));
        shape_inference::ShapeHandle jj2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &jj2_shape));
        shape_inference::ShapeHandle vv2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &vv2_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &d_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("PushMatricesGrad")

.Input("flag : int64")
.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("d : int64")
.Output("grad_ii1 : int64")
.Output("grad_jj1 : int64")
.Output("grad_vv1 : double")
.Output("grad_ii2 : int64")
.Output("grad_jj2 : int64")
.Output("grad_vv2 : double")
.Output("grad_d : int64");


class PushMatricesOp : public OpKernel {
private:
  
public:
  explicit PushMatricesOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& ii2 = context->input(3);
    const Tensor& jj2 = context->input(4);
    const Tensor& vv2 = context->input(5);
    const Tensor& d = context->input(6);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n1 = ii1_shape.dim_size(0);
    int n2 = ii2_shape.dim_size(0);
    TensorShape flag_shape({});
            
    // create output tensor
    
    Tensor* flag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, flag_shape, &flag));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto flag_tensor = flag->flat<int64>().data();   

    // implement your forward function here 

    // TODO:
    // printf("n1 = %d, n2 = %d\n", n1, n2);
    push_matrix(flag_tensor, ii1_tensor, jj1_tensor, vv1_tensor, n1, ii2_tensor, jj2_tensor, n2, *d_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("PushMatrices").Device(DEVICE_CPU), PushMatricesOp);
