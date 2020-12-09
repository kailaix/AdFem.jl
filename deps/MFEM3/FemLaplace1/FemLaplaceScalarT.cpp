#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemLaplaceScalarT.h"


REGISTER_OP("FemLaplaceScalarT")
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

REGISTER_OP("FemLaplaceScalarTGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("kappa : double")
.Output("grad_kappa : double");

/*-------------------------------------------------------------------------------------*/

class FemLaplaceScalarTOp : public OpKernel {
private:
  
public:
  explicit FemLaplaceScalarTOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& kappa = context->input(0);
    
    
    const TensorShape& kappa_shape = kappa.shape();
    
    
    DCHECK_EQ(kappa_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({mmesh3.elem_ndof * mmesh3.elem_ndof * mmesh3.ngauss,2});
    TensorShape vv_shape({mmesh3.elem_ndof * mmesh3.elem_ndof * mmesh3.ngauss});
      
            
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
    MFEM::FemLaplaceScalarT_forward(indices_tensor, vv_tensor, kappa_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemLaplaceScalarT").Device(DEVICE_CPU), FemLaplaceScalarTOp);



class FemLaplaceScalarTGradOp : public OpKernel {
private:
  
public:
  explicit FemLaplaceScalarTGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
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
     MFEM::FemLaplaceScalarT_backward(
      grad_kappa_tensor, grad_vv_tensor, indices_tensor, vv_tensor, kappa_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemLaplaceScalarTGrad").Device(DEVICE_CPU), FemLaplaceScalarTGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("FemLaplaceScalarTGpu")
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

REGISTER_OP("FemLaplaceScalarTGpuGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("kappa : double")
.Output("grad_kappa : double");

class FemLaplaceScalarTOpGPU : public OpKernel {
private:
  
public:
  explicit FemLaplaceScalarTOpGPU(OpKernelConstruction* context) : OpKernel(context) {

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
REGISTER_KERNEL_BUILDER(Name("FemLaplaceScalarTGpu").Device(DEVICE_GPU), FemLaplaceScalarTOpGPU);

class FemLaplaceScalarTGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemLaplaceScalarTGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
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
REGISTER_KERNEL_BUILDER(Name("FemLaplaceScalarTGpuGrad").Device(DEVICE_GPU), FemLaplaceScalarTGradOpGPU);

#endif