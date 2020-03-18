#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "RateStateFrictionBS.h"


REGISTER_OP("RateStateFriction")

.Input("a : double")
.Input("uold : double")
.Input("v0 : double")
.Input("psi : double")
.Input("sigmazx : double")
.Input("sigmazy : double")
.Input("eta : double")
.Input("deltat : double")
.Output("u : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle uold_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &uold_shape));
        shape_inference::ShapeHandle v0_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &v0_shape));
        shape_inference::ShapeHandle psi_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &psi_shape));
        shape_inference::ShapeHandle sigmazx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &sigmazx_shape));
        shape_inference::ShapeHandle sigmazy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &sigmazy_shape));
        shape_inference::ShapeHandle eta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &eta_shape));
        shape_inference::ShapeHandle deltat_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &deltat_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("RateStateFrictionGrad")

.Input("grad_u : double")
.Input("u : double")
.Input("a : double")
.Input("uold : double")
.Input("v0 : double")
.Input("psi : double")
.Input("sigmazx : double")
.Input("sigmazy : double")
.Input("eta : double")
.Input("deltat : double")
.Output("grad_a : double")
.Output("grad_uold : double")
.Output("grad_v0 : double")
.Output("grad_psi : double")
.Output("grad_sigmazx : double")
.Output("grad_sigmazy : double")
.Output("grad_eta : double")
.Output("grad_deltat : double");


class RateStateFrictionOp : public OpKernel {
private:
  
public:
  explicit RateStateFrictionOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(8, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& uold = context->input(1);
    const Tensor& v0 = context->input(2);
    const Tensor& psi = context->input(3);
    const Tensor& sigmazx = context->input(4);
    const Tensor& sigmazy = context->input(5);
    const Tensor& eta = context->input(6);
    const Tensor& deltat = context->input(7);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& uold_shape = uold.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigmazx_shape = sigmazx.shape();
    const TensorShape& sigmazy_shape = sigmazy.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& deltat_shape = deltat.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(uold_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigmazx_shape.dims(), 1);
    DCHECK_EQ(sigmazy_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);
    DCHECK_EQ(deltat_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape u_shape({n});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto uold_tensor = uold.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigmazx_tensor = sigmazx.flat<double>().data();
    auto sigmazy_tensor = sigmazy.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto deltat_tensor = deltat.flat<double>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    forward(u_tensor, a_tensor, uold_tensor, *v0_tensor, psi_tensor, sigmazx_tensor, sigmazy_tensor, 
        *eta_tensor, *deltat_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("RateStateFriction").Device(DEVICE_CPU), RateStateFrictionOp);


class RateStateFrictionGradOp : public OpKernel {
private:
  
public:
  explicit RateStateFrictionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& uold = context->input(3);
    const Tensor& v0 = context->input(4);
    const Tensor& psi = context->input(5);
    const Tensor& sigmazx = context->input(6);
    const Tensor& sigmazy = context->input(7);
    const Tensor& eta = context->input(8);
    const Tensor& deltat = context->input(9);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& uold_shape = uold.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigmazx_shape = sigmazx.shape();
    const TensorShape& sigmazy_shape = sigmazy.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& deltat_shape = deltat.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(uold_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigmazx_shape.dims(), 1);
    DCHECK_EQ(sigmazy_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);
    DCHECK_EQ(deltat_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_uold_shape(uold_shape);
    TensorShape grad_v0_shape(v0_shape);
    TensorShape grad_psi_shape(psi_shape);
    TensorShape grad_sigmazx_shape(sigmazx_shape);
    TensorShape grad_sigmazy_shape(sigmazy_shape);
    TensorShape grad_eta_shape(eta_shape);
    TensorShape grad_deltat_shape(deltat_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_uold = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_uold_shape, &grad_uold));
    Tensor* grad_v0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_v0_shape, &grad_v0));
    Tensor* grad_psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_psi_shape, &grad_psi));
    Tensor* grad_sigmazx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_sigmazx_shape, &grad_sigmazx));
    Tensor* grad_sigmazy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_sigmazy_shape, &grad_sigmazy));
    Tensor* grad_eta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_eta_shape, &grad_eta));
    Tensor* grad_deltat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_deltat_shape, &grad_deltat));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto uold_tensor = uold.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigmazx_tensor = sigmazx.flat<double>().data();
    auto sigmazy_tensor = sigmazy.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto deltat_tensor = deltat.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_uold_tensor = grad_uold->flat<double>().data();
    auto grad_v0_tensor = grad_v0->flat<double>().data();
    auto grad_psi_tensor = grad_psi->flat<double>().data();
    auto grad_sigmazx_tensor = grad_sigmazx->flat<double>().data();
    auto grad_sigmazy_tensor = grad_sigmazy->flat<double>().data();
    auto grad_eta_tensor = grad_eta->flat<double>().data();
    auto grad_deltat_tensor = grad_deltat->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = a_shape.dim_size(0);
    backward(
      grad_a_tensor, grad_uold_tensor, grad_psi_tensor, grad_sigmazx_tensor, grad_sigmazy_tensor, grad_u_tensor,
      u_tensor, a_tensor, uold_tensor, *v0_tensor, psi_tensor, sigmazx_tensor, sigmazy_tensor, 
        *eta_tensor, *deltat_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("RateStateFrictionGrad").Device(DEVICE_CPU), RateStateFrictionGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class RateStateFrictionOpGPU : public OpKernel {
private:
  
public:
  explicit RateStateFrictionOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(8, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& uold = context->input(1);
    const Tensor& v0 = context->input(2);
    const Tensor& psi = context->input(3);
    const Tensor& sigmazx = context->input(4);
    const Tensor& sigmazy = context->input(5);
    const Tensor& eta = context->input(6);
    const Tensor& deltat = context->input(7);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& uold_shape = uold.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigmazx_shape = sigmazx.shape();
    const TensorShape& sigmazy_shape = sigmazy.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& deltat_shape = deltat.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(uold_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigmazx_shape.dims(), 1);
    DCHECK_EQ(sigmazy_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);
    DCHECK_EQ(deltat_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape u_shape({-1});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto uold_tensor = uold.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigmazx_tensor = sigmazx.flat<double>().data();
    auto sigmazy_tensor = sigmazy.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto deltat_tensor = deltat.flat<double>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("RateStateFriction").Device(DEVICE_GPU), RateStateFrictionOpGPU);

class RateStateFrictionGradOpGPU : public OpKernel {
private:
  
public:
  explicit RateStateFrictionGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& uold = context->input(3);
    const Tensor& v0 = context->input(4);
    const Tensor& psi = context->input(5);
    const Tensor& sigmazx = context->input(6);
    const Tensor& sigmazy = context->input(7);
    const Tensor& eta = context->input(8);
    const Tensor& deltat = context->input(9);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& uold_shape = uold.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigmazx_shape = sigmazx.shape();
    const TensorShape& sigmazy_shape = sigmazy.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& deltat_shape = deltat.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(uold_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigmazx_shape.dims(), 1);
    DCHECK_EQ(sigmazy_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);
    DCHECK_EQ(deltat_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_uold_shape(uold_shape);
    TensorShape grad_v0_shape(v0_shape);
    TensorShape grad_psi_shape(psi_shape);
    TensorShape grad_sigmazx_shape(sigmazx_shape);
    TensorShape grad_sigmazy_shape(sigmazy_shape);
    TensorShape grad_eta_shape(eta_shape);
    TensorShape grad_deltat_shape(deltat_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_uold = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_uold_shape, &grad_uold));
    Tensor* grad_v0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_v0_shape, &grad_v0));
    Tensor* grad_psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_psi_shape, &grad_psi));
    Tensor* grad_sigmazx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_sigmazx_shape, &grad_sigmazx));
    Tensor* grad_sigmazy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_sigmazy_shape, &grad_sigmazy));
    Tensor* grad_eta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_eta_shape, &grad_eta));
    Tensor* grad_deltat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_deltat_shape, &grad_deltat));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto uold_tensor = uold.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigmazx_tensor = sigmazx.flat<double>().data();
    auto sigmazy_tensor = sigmazy.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto deltat_tensor = deltat.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_uold_tensor = grad_uold->flat<double>().data();
    auto grad_v0_tensor = grad_v0->flat<double>().data();
    auto grad_psi_tensor = grad_psi->flat<double>().data();
    auto grad_sigmazx_tensor = grad_sigmazx->flat<double>().data();
    auto grad_sigmazy_tensor = grad_sigmazy->flat<double>().data();
    auto grad_eta_tensor = grad_eta->flat<double>().data();
    auto grad_deltat_tensor = grad_deltat->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("RateStateFrictionGrad").Device(DEVICE_GPU), RateStateFrictionGradOpGPU);

#endif