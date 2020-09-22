#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeFemStiffnessMatrixMfem.h"


REGISTER_OP("ComputeFemStiffnessMatrixMfem")
.Input("kappa : double")
.Output("indices : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle kappa_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &kappa_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemStiffnessMatrixMfemGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("kappa : double")
.Output("grad_kappa : double");

/*-------------------------------------------------------------------------------------*/

class ComputeFemStiffnessMatrixMfemOp : public OpKernel {
private:
  
public:
  explicit ComputeFemStiffnessMatrixMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& kappa = context->input(0);
    
    
    const TensorShape& kappa_shape = kappa.shape();
    
    
    DCHECK_EQ(kappa_shape.dims(), 1);

    // extra check
        
    // create output shape
    int N = (mmesh.elem_ndof * 2) * (mmesh.elem_ndof * 2) * mmesh.ngauss;
    TensorShape indices_shape({N,2});
    TensorShape vv_shape({N});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto kappa_tensor = kappa.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::ComputeFemStiffnessMatrixMfem_forward(indices_tensor, vv_tensor, kappa_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemStiffnessMatrixMfem").Device(DEVICE_CPU), ComputeFemStiffnessMatrixMfemOp);



class ComputeFemStiffnessMatrixMfemGradOp : public OpKernel {
private:
  
public:
  explicit ComputeFemStiffnessMatrixMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& kappa = context->input(3);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& kappa_shape = kappa.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(kappa_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_kappa_shape(kappa_shape);
            
    // create output tensor
    
    Tensor* grad_kappa = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_kappa_shape, &grad_kappa));
    
    // get the corresponding Eigen tensors for data access
    
    auto kappa_tensor = kappa.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_kappa_tensor = grad_kappa->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_kappa->flat<double>().setZero();
    MFEM::ComputeFemStiffnessMatrixMfem_backward(grad_kappa_tensor, grad_vv_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemStiffnessMatrixMfemGrad").Device(DEVICE_CPU), ComputeFemStiffnessMatrixMfemGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class ComputeFemStiffnessMatrixMfemOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemStiffnessMatrixMfemOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& kappa = context->input(0);
    
    
    const TensorShape& kappa_shape = kappa.shape();
    
    
    DCHECK_EQ(kappa_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({-1,2});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto kappa_tensor = kappa.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemStiffnessMatrixMfem").Device(DEVICE_GPU), ComputeFemStiffnessMatrixMfemOpGPU);

class ComputeFemStiffnessMatrixMfemGradOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemStiffnessMatrixMfemGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& kappa = context->input(3);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& kappa_shape = kappa.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(kappa_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_kappa_shape(kappa_shape);
            
    // create output tensor
    
    Tensor* grad_kappa = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_kappa_shape, &grad_kappa));
    
    // get the corresponding Eigen tensors for data access
    
    auto kappa_tensor = kappa.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_kappa_tensor = grad_kappa->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemStiffnessMatrixMfemGrad").Device(DEVICE_GPU), ComputeFemStiffnessMatrixMfemGradOpGPU);

#endif