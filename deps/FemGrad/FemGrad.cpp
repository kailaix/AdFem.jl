#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemGrad.h"


REGISTER_OP("FemGrad")
.Input("u : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad : double")
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

REGISTER_OP("FemGradGrad")
.Input("grad_grad : double")
.Input("grad : double")
.Input("u : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class FemGradOp : public OpKernel {
private:
  
public:
  explicit FemGradOp(OpKernelConstruction* context) : OpKernel(context) {

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
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();

    int m_ = *m_tensor, n_ = *n_tensor;
    

    TensorShape grad_shape({4*m_*n_,2});
            
    // create output tensor
    
    Tensor* grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_shape, &grad));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_tensor = grad->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    FemGrad_forward(grad_tensor, u_tensor, m_, n_, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemGrad").Device(DEVICE_CPU), FemGradOp);



class FemGradGradOp : public OpKernel {
private:
  
public:
  explicit FemGradGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_grad = context->input(0);
    const Tensor& grad = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_grad_shape = grad_grad.shape();
    const TensorShape& grad_shape = grad.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_grad_shape.dims(), 2);
    DCHECK_EQ(grad_shape.dims(), 2);
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
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_grad_tensor = grad_grad.flat<double>().data();
    auto grad_tensor = grad.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 
    grad_u->flat<double>().setZero();

    // TODO:
    FemGrad_backward(grad_u_tensor, grad_grad_tensor, u_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemGradGrad").Device(DEVICE_CPU), FemGradGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class FemGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {

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
    
    TensorShape grad_shape({-1,2});
            
    // create output tensor
    
    Tensor* grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_shape, &grad));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_tensor = grad->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemGrad").Device(DEVICE_GPU), FemGradOpGPU);

class FemGradGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemGradGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_grad = context->input(0);
    const Tensor& grad = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_grad_shape = grad_grad.shape();
    const TensorShape& grad_shape = grad.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_grad_shape.dims(), 2);
    DCHECK_EQ(grad_shape.dims(), 2);
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
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_grad_tensor = grad_grad.flat<double>().data();
    auto grad_tensor = grad.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemGradGrad").Device(DEVICE_GPU), FemGradGradOpGPU);

#endif