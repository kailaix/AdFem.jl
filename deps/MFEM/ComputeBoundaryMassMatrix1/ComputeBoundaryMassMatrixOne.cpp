#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeBoundaryMassMatrixOne.h"


REGISTER_OP("ComputeBoundaryMassMatrixOne")
.Input("c : double")
.Input("idx : int64")
.Output("ij : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &c_shape));
        shape_inference::ShapeHandle idx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &idx_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeBoundaryMassMatrixOneGrad")
.Input("grad_vv : double")
.Input("ij : int64")
.Input("vv : double")
.Input("c : double")
.Input("idx : int64")
.Output("grad_c : double")
.Output("grad_idx : int64");

/*-------------------------------------------------------------------------------------*/

class ComputeBoundaryMassMatrixOneOp : public OpKernel {
private:
  
public:
  explicit ComputeBoundaryMassMatrixOneOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& c = context->input(0);
    const Tensor& idx = context->input(1);
    
    
    const TensorShape& c_shape = c.shape();
    const TensorShape& idx_shape = idx.shape();
    
    
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(idx_shape.dims(), 2);

    // extra check
        
    // create output shape
    int nedge = idx_shape.dim_size(0);
    int N = LineIntegralN * 4 * nedge;
    
    TensorShape ij_shape({N,2});
    TensorShape vv_shape({N});
            
    // create output tensor
    
    Tensor* ij = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ij_shape, &ij));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto c_tensor = c.flat<double>().data();
    auto idx_tensor = idx.flat<int64>().data();
    auto ij_tensor = ij->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
      MFEM::ComputeBoundaryMassMatrixOneForward(
        ij_tensor, vv_tensor, c_tensor, idx_tensor, nedge);


  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeBoundaryMassMatrixOne").Device(DEVICE_CPU), ComputeBoundaryMassMatrixOneOp);



class ComputeBoundaryMassMatrixOneGradOp : public OpKernel {
private:
  
public:
  explicit ComputeBoundaryMassMatrixOneGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ij = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& c = context->input(3);
    const Tensor& idx = context->input(4);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ij_shape = ij.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& idx_shape = idx.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ij_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(idx_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_idx_shape(idx_shape);
            
    // create output tensor
    
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_c_shape, &grad_c));
    Tensor* grad_idx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_idx_shape, &grad_idx));
    
    // get the corresponding Eigen tensors for data access
    
    auto c_tensor = c.flat<double>().data();
    auto idx_tensor = idx.flat<int64>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ij_tensor = ij.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int nedge = idx_shape.dim_size(0);
    MFEM::ComputeBoundaryMassMatrixOneBackward(
      grad_c_tensor, grad_vv_tensor, vv_tensor, c_tensor, idx_tensor, nedge);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeBoundaryMassMatrixOneGrad").Device(DEVICE_CPU), ComputeBoundaryMassMatrixOneGradOp);

