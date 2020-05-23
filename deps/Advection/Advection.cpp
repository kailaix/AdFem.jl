#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "Advection.h"


REGISTER_OP("Advection")
.Input("v : double")
.Input("u : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("a : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &v_shape));
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &u_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("AdvectionGrad")
.Input("grad_a : double")
.Input("a : double")
.Input("v : double")
.Input("u : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_v : double")
.Output("grad_u : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class AdvectionOp : public OpKernel {
private:
  
public:
  explicit AdvectionOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& v = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& h = context->input(4);
    
    
    const TensorShape& v_shape = v.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(v_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();


    TensorShape a_shape({*m_tensor * *n_tensor});
            
    // create output tensor
    
    Tensor* a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a_shape, &a));
    
    // get the corresponding Eigen tensors for data access
    
    auto v_tensor = v.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    
    auto h_tensor = h.flat<double>().data();
    auto a_tensor = a->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(a_tensor, v_tensor, u_tensor, *m_tensor, *n_tensor, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("Advection").Device(DEVICE_CPU), AdvectionOp);



class AdvectionGradOp : public OpKernel {
private:
  
public:
  explicit AdvectionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_a = context->input(0);
    const Tensor& a = context->input(1);
    const Tensor& v = context->input(2);
    const Tensor& u = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& h = context->input(6);
    
    
    const TensorShape& grad_a_shape = grad_a.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_a_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_v_shape(v_shape);
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_v_shape, &grad_v));
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_u_shape, &grad_u));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto v_tensor = v.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_a_tensor = grad_a.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(
      grad_v_tensor, grad_u_tensor, grad_a_tensor, 
      a_tensor, v_tensor, u_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("AdvectionGrad").Device(DEVICE_CPU), AdvectionGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class AdvectionOpGPU : public OpKernel {
private:
  
public:
  explicit AdvectionOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& v = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& h = context->input(4);
    
    
    const TensorShape& v_shape = v.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(v_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape a_shape({-1});
            
    // create output tensor
    
    Tensor* a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a_shape, &a));
    
    // get the corresponding Eigen tensors for data access
    
    auto v_tensor = v.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto a_tensor = a->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Advection").Device(DEVICE_GPU), AdvectionOpGPU);

class AdvectionGradOpGPU : public OpKernel {
private:
  
public:
  explicit AdvectionGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_a = context->input(0);
    const Tensor& a = context->input(1);
    const Tensor& v = context->input(2);
    const Tensor& u = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& h = context->input(6);
    
    
    const TensorShape& grad_a_shape = grad_a.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_a_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_v_shape(v_shape);
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_v_shape, &grad_v));
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_u_shape, &grad_u));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto v_tensor = v.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_a_tensor = grad_a.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("AdvectionGrad").Device(DEVICE_GPU), AdvectionGradOpGPU);

#endif