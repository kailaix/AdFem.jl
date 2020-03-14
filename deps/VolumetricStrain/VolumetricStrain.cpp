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
#include "VolumetricStrain.h"


REGISTER_OP("VolumetricStrain")

.Input("u : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("eps : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("VolumetricStrainGrad")

.Input("grad_eps : double")
.Input("eps : double")
.Input("u : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_m : int32")
.Output("grad_n : int32")
.Output("grad_h : double");


class VolumetricStrainOp : public OpKernel {
private:
  
public:
  explicit VolumetricStrainOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);
    

    // extra check
        
    // create output shape

    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    int m_ = *m_tensor, n_ = *n_tensor;
    
    DCHECK_EQ(u_shape.dim_size(0), (m_+1)*(n_+1)*2);
    TensorShape eps_shape({m_*n_});
            
    // create output tensor
    
    Tensor* eps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, eps_shape, &eps));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    
    auto h_tensor = h.flat<double>().data();
    auto eps_tensor = eps->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(eps_tensor, u_tensor, m_, n_, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("VolumetricStrain").Device(DEVICE_CPU), VolumetricStrainOp);



class VolumetricStrainGradOp : public OpKernel {
private:
  
public:
  explicit VolumetricStrainGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_eps = context->input(0);
    const Tensor& eps = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_eps_shape = grad_eps.shape();
    const TensorShape& eps_shape = eps.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_eps_shape.dims(), 1);
    DCHECK_EQ(eps_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    // Tensor* grad_m = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    // Tensor* grad_n = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    // Tensor* grad_h = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_eps_tensor = grad_eps.flat<double>().data();
    auto eps_tensor = eps.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    // auto grad_m_tensor = grad_m->flat<int32>().data();
    // auto grad_n_tensor = grad_n->flat<int32>().data();
    // auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    int m_ = *m_tensor, n_ = *n_tensor;
    backward(grad_u_tensor, grad_eps_tensor, eps_tensor, u_tensor, m_, n_, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("VolumetricStrainGrad").Device(DEVICE_CPU), VolumetricStrainGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class VolumetricStrainOpGPU : public OpKernel {
private:
  
public:
  explicit VolumetricStrainOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape eps_shape({-1});
            
    // create output tensor
    
    Tensor* eps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, eps_shape, &eps));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto eps_tensor = eps->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("VolumetricStrain").Device(DEVICE_GPU), VolumetricStrainOpGPU);

class VolumetricStrainGradOpGPU : public OpKernel {
private:
  
public:
  explicit VolumetricStrainGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_eps = context->input(0);
    const Tensor& eps = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_eps_shape = grad_eps.shape();
    const TensorShape& eps_shape = eps.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_eps_shape.dims(), 1);
    DCHECK_EQ(eps_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_eps_tensor = grad_eps.flat<double>().data();
    auto eps_tensor = eps.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("VolumetricStrainGrad").Device(DEVICE_GPU), VolumetricStrainGradOpGPU);

#endif