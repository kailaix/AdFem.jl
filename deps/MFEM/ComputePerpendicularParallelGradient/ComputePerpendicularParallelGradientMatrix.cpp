#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputePerpendicularParallelGradientMatrix.h"


REGISTER_OP("ComputePerpendicularParallelGradientMatrix")
.Input("nv : double")
.Input("cmat : double")
.Input("left : int64")
.Input("right : int64")
.Output("nmat : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle nv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &nv_shape));
        shape_inference::ShapeHandle cmat_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &cmat_shape));
        shape_inference::ShapeHandle left_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &left_shape));
        shape_inference::ShapeHandle right_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &right_shape));

        c->set_output(0, c->MakeShape({-1,2,2}));
    return Status::OK();
  });

REGISTER_OP("ComputePerpendicularParallelGradientMatrixGrad")
.Input("grad_nmat : double")
.Input("nmat : double")
.Input("nv : double")
.Input("cmat : double")
.Input("left : int64")
.Input("right : int64")
.Output("grad_nv : double")
.Output("grad_cmat : double")
.Output("grad_left : int64")
.Output("grad_right : int64");

/*-------------------------------------------------------------------------------------*/

class ComputePerpendicularParallelGradientMatrixOp : public OpKernel {
private:
  
public:
  explicit ComputePerpendicularParallelGradientMatrixOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& nv = context->input(0);
    const Tensor& cmat = context->input(1);
    const Tensor& left = context->input(2);
    const Tensor& right = context->input(3);
    
    
    const TensorShape& nv_shape = nv.shape();
    const TensorShape& cmat_shape = cmat.shape();
    const TensorShape& left_shape = left.shape();
    const TensorShape& right_shape = right.shape();
    
    
    DCHECK_EQ(nv_shape.dims(), 2);
    DCHECK_EQ(cmat_shape.dims(), 3);
    DCHECK_EQ(left_shape.dims(), 0);
    DCHECK_EQ(right_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape nmat_shape({mmesh.ngauss,2,2});
            
    // create output tensor
    
    Tensor* nmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, nmat_shape, &nmat));
    
    // get the corresponding Eigen tensors for data access
    
    auto nv_tensor = nv.flat<double>().data();
    auto cmat_tensor = cmat.flat<double>().data();
    auto left_tensor = left.flat<int64>().data();
    auto right_tensor = right.flat<int64>().data();
    auto nmat_tensor = nmat->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::ComputePerpendicularParallelGradientForward(
      nmat_tensor, nv_tensor, cmat_tensor, left_tensor, right_tensor);


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePerpendicularParallelGradientMatrix").Device(DEVICE_CPU), ComputePerpendicularParallelGradientMatrixOp);



class ComputePerpendicularParallelGradientMatrixGradOp : public OpKernel {
private:
  
public:
  explicit ComputePerpendicularParallelGradientMatrixGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_nmat = context->input(0);
    const Tensor& nmat = context->input(1);
    const Tensor& nv = context->input(2);
    const Tensor& cmat = context->input(3);
    const Tensor& left = context->input(4);
    const Tensor& right = context->input(5);
    
    
    const TensorShape& grad_nmat_shape = grad_nmat.shape();
    const TensorShape& nmat_shape = nmat.shape();
    const TensorShape& nv_shape = nv.shape();
    const TensorShape& cmat_shape = cmat.shape();
    const TensorShape& left_shape = left.shape();
    const TensorShape& right_shape = right.shape();
    
    
    DCHECK_EQ(grad_nmat_shape.dims(), 3);
    DCHECK_EQ(nmat_shape.dims(), 3);
    DCHECK_EQ(nv_shape.dims(), 2);
    DCHECK_EQ(cmat_shape.dims(), 3);
    DCHECK_EQ(left_shape.dims(), 0);
    DCHECK_EQ(right_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_nv_shape(nv_shape);
    TensorShape grad_cmat_shape(cmat_shape);
    TensorShape grad_left_shape(left_shape);
    TensorShape grad_right_shape(right_shape);
            
    // create output tensor
    
    Tensor* grad_nv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_nv_shape, &grad_nv));
    Tensor* grad_cmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_cmat_shape, &grad_cmat));
    Tensor* grad_left = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_left_shape, &grad_left));
    Tensor* grad_right = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_right_shape, &grad_right));
    
    // get the corresponding Eigen tensors for data access
    
    auto nv_tensor = nv.flat<double>().data();
    auto cmat_tensor = cmat.flat<double>().data();
    auto left_tensor = left.flat<int64>().data();
    auto right_tensor = right.flat<int64>().data();
    auto grad_nmat_tensor = grad_nmat.flat<double>().data();
    auto nmat_tensor = nmat.flat<double>().data();
    auto grad_nv_tensor = grad_nv->flat<double>().data();
    auto grad_cmat_tensor = grad_cmat->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    MFEM::ComputePerpendicularParallelGradientBackward(
      grad_cmat_tensor, grad_nmat_tensor, nmat_tensor, nv_tensor, cmat_tensor, left_tensor, right_tensor);
    
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputePerpendicularParallelGradientMatrixGrad").Device(DEVICE_CPU), ComputePerpendicularParallelGradientMatrixGradOp);

