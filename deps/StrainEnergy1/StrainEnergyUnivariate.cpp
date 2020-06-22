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
#include "StrainEnergyUnivariate.h"


REGISTER_OP("StrainEnergyUnivariate")

.Input("sigma : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &sigma_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("StrainEnergyUnivariateGrad")

.Input("grad_out : double")
.Input("out : double")
.Input("sigma : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("grad_sigma : double")
.Output("grad_m : int32")
.Output("grad_n : int32")
.Output("grad_h : double");


class StrainEnergyUnivariateOp : public OpKernel {
private:
  
public:
  explicit StrainEnergyUnivariateOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& sigma = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape

    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();

    int m_ = *m_tensor, n_ = *n_tensor;
    
    TensorShape out_shape({(m_+1)*(n_+1)});

    DCHECK_EQ(sigma_shape.dim_size(0), 4*m_*n_);
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 
    SE1_forward(out_tensor, sigma_tensor, m_, n_, *h_tensor);

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("StrainEnergyUnivariate").Device(DEVICE_CPU), StrainEnergyUnivariateOp);



class StrainEnergyUnivariateGradOp : public OpKernel {
private:
  
public:
  explicit StrainEnergyUnivariateGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& sigma = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigma_shape, &grad_sigma));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int m_ = *m_tensor, n_ = *n_tensor;


    SE1_backward(grad_sigma_tensor, grad_out_tensor, out_tensor, sigma_tensor, m_, n_, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("StrainEnergyUnivariateGrad").Device(DEVICE_CPU), StrainEnergyUnivariateGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class StrainEnergyUnivariateOpGPU : public OpKernel {
private:
  
public:
  explicit StrainEnergyUnivariateOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& sigma = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("StrainEnergyUnivariate").Device(DEVICE_GPU), StrainEnergyUnivariateOpGPU);

class StrainEnergyUnivariateGradOpGPU : public OpKernel {
private:
  
public:
  explicit StrainEnergyUnivariateGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& sigma = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigma_shape, &grad_sigma));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("StrainEnergyUnivariateGrad").Device(DEVICE_GPU), StrainEnergyUnivariateGradOpGPU);

#endif