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
#include "ComputeVel.h"


REGISTER_OP("ComputeVel")

.Input("a : double")
.Input("v0 : double")
.Input("psi : double")
.Input("sigma : double")
.Input("tau : double")
.Input("eta : double")
.Output("v : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle v0_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &v0_shape));
        shape_inference::ShapeHandle psi_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &psi_shape));
        shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sigma_shape));
        shape_inference::ShapeHandle tau_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &tau_shape));
        shape_inference::ShapeHandle eta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &eta_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeVelGrad")

.Input("grad_v : double")
.Input("v : double")
.Input("a : double")
.Input("v0 : double")
.Input("psi : double")
.Input("sigma : double")
.Input("tau : double")
.Input("eta : double")
.Output("grad_a : double")
.Output("grad_v0 : double")
.Output("grad_psi : double")
.Output("grad_sigma : double")
.Output("grad_tau : double")
.Output("grad_eta : double");


class ComputeVelOp : public OpKernel {
private:
  
public:
  explicit ComputeVelOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& v0 = context->input(1);
    const Tensor& psi = context->input(2);
    const Tensor& sigma = context->input(3);
    const Tensor& tau = context->input(4);
    const Tensor& eta = context->input(5);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& eta_shape = eta.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape v_shape({n});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(v_tensor, a_tensor, *v0_tensor, psi_tensor, sigma_tensor, 
     tau_tensor, *eta_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeVel").Device(DEVICE_CPU), ComputeVelOp);



class ComputeVelGradOp : public OpKernel {
private:
  
public:
  explicit ComputeVelGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& v0 = context->input(3);
    const Tensor& psi = context->input(4);
    const Tensor& sigma = context->input(5);
    const Tensor& tau = context->input(6);
    const Tensor& eta = context->input(7);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& eta_shape = eta.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    int n = a_shape.dim_size(0);
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_v0_shape(v0_shape);
    TensorShape grad_psi_shape(psi_shape);
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_tau_shape(tau_shape);
    TensorShape grad_eta_shape(eta_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_v0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v0_shape, &grad_v0));
    Tensor* grad_psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_psi_shape, &grad_psi));
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_sigma_shape, &grad_sigma));
    Tensor* grad_tau = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_tau_shape, &grad_tau));
    Tensor* grad_eta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_eta_shape, &grad_eta));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_v0_tensor = grad_v0->flat<double>().data();
    auto grad_psi_tensor = grad_psi->flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_tau_tensor = grad_tau->flat<double>().data();
    auto grad_eta_tensor = grad_eta->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    backward(grad_a_tensor, grad_psi_tensor, grad_sigma_tensor, grad_tau_tensor, grad_eta_tensor,
        grad_v0_tensor, grad_v_tensor, v_tensor, a_tensor, *v0_tensor, psi_tensor, 
        sigma_tensor, tau_tensor, *eta_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeVelGrad").Device(DEVICE_CPU), ComputeVelGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class ComputeVelOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeVelOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& v0 = context->input(1);
    const Tensor& psi = context->input(2);
    const Tensor& sigma = context->input(3);
    const Tensor& tau = context->input(4);
    const Tensor& eta = context->input(5);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& eta_shape = eta.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape v_shape({-1});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeVel").Device(DEVICE_GPU), ComputeVelOpGPU);

class ComputeVelGradOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeVelGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& v0 = context->input(3);
    const Tensor& psi = context->input(4);
    const Tensor& sigma = context->input(5);
    const Tensor& tau = context->input(6);
    const Tensor& eta = context->input(7);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& v0_shape = v0.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& eta_shape = eta.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(v0_shape.dims(), 0);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_v0_shape(v0_shape);
    TensorShape grad_psi_shape(psi_shape);
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_tau_shape(tau_shape);
    TensorShape grad_eta_shape(eta_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_v0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v0_shape, &grad_v0));
    Tensor* grad_psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_psi_shape, &grad_psi));
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_sigma_shape, &grad_sigma));
    Tensor* grad_tau = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_tau_shape, &grad_tau));
    Tensor* grad_eta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_eta_shape, &grad_eta));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto v0_tensor = v0.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_v0_tensor = grad_v0->flat<double>().data();
    auto grad_psi_tensor = grad_psi->flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_tau_tensor = grad_tau->flat<double>().data();
    auto grad_eta_tensor = grad_eta->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeVelGrad").Device(DEVICE_GPU), ComputeVelGradOpGPU);

#endif