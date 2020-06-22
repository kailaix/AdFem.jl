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
#include "StrainOpUnivariate.h"


REGISTER_OP("StrainOpUnivariate")

.Input("u : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("strain : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Matrix(-1,2));
    return Status::OK();
  });

REGISTER_OP("StrainOpUnivariateGrad")

.Input("grad_strain : double")
.Input("strain : double")
.Input("u : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_m : int32")
.Output("grad_n : int32")
.Output("grad_h : double");


class StrainOpUnivariateOp : public OpKernel {
private:
  
public:
  explicit StrainOpUnivariateOp(OpKernelConstruction* context) : OpKernel(context) {

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
    TensorShape strain_shape({ 4 * (*m_tensor) * (*n_tensor), 2});
            
    // create output tensor
    
    Tensor* strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, strain_shape, &strain));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    
    auto h_tensor = h.flat<double>().data();
    auto strain_tensor = strain->flat<double>().data();   

    // implement your forward function here 
    SO1_forward(strain_tensor, u_tensor, *m_tensor, *n_tensor, *h_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("StrainOpUnivariate").Device(DEVICE_CPU), StrainOpUnivariateOp);



class StrainOpUnivariateGradOp : public OpKernel {
private:
  
public:
  explicit StrainOpUnivariateGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_strain = context->input(0);
    const Tensor& strain = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_strain_shape = grad_strain.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_strain_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
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
    auto grad_strain_tensor = grad_strain.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    SO1_backward( 
      grad_u_tensor, grad_strain_tensor,
      strain_tensor, u_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("StrainOpUnivariateGrad").Device(DEVICE_CPU), StrainOpUnivariateGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class StrainOpUnivariateOpGPU : public OpKernel {
private:
  
public:
  explicit StrainOpUnivariateOpGPU(OpKernelConstruction* context) : OpKernel(context) {

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
    
    TensorShape strain_shape({-1,2});
            
    // create output tensor
    
    Tensor* strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, strain_shape, &strain));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto strain_tensor = strain->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("StrainOpUnivariate").Device(DEVICE_GPU), StrainOpUnivariateOpGPU);

class StrainOpUnivariateGradOpGPU : public OpKernel {
private:
  
public:
  explicit StrainOpUnivariateGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_strain = context->input(0);
    const Tensor& strain = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_strain_shape = grad_strain.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_strain_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
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
    auto grad_strain_tensor = grad_strain.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("StrainOpUnivariateGrad").Device(DEVICE_GPU), StrainOpUnivariateGradOpGPU);

#endif