#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemSource.h"


REGISTER_OP("FemSource")
.Input("f : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("rhs : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &f_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FemSourceGrad")
.Input("grad_rhs : double")
.Input("rhs : double")
.Input("f : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_f : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class FemSourceOp : public OpKernel {
private:
  
public:
  explicit FemSourceOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& f = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& f_shape = f.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    int m_ = *m.flat<int64>().data(), n_ = *n.flat<int64>().data();
    TensorShape rhs_shape({(m_+1)*(n_+1)});

            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    rhs->flat<double>().setZero();
    FemSource_forward(rhs_tensor, f_tensor, *m_tensor, *n_tensor, *h_tensor);


  }
};
REGISTER_KERNEL_BUILDER(Name("FemSource").Device(DEVICE_CPU), FemSourceOp);



class FemSourceGradOp : public OpKernel {
private:
  
public:
  explicit FemSourceGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_rhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& f = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_f_shape, &grad_f));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_f->flat<double>().setZero();
    FemSource_backward(grad_f_tensor, grad_rhs_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemSourceGrad").Device(DEVICE_CPU), FemSourceGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class FemSourceOpGPU : public OpKernel {
private:
  
public:
  explicit FemSourceOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& f = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& f_shape = f.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape rhs_shape({-1});
            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemSource").Device(DEVICE_GPU), FemSourceOpGPU);

class FemSourceGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemSourceGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_rhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& f = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_f_shape, &grad_f));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemSourceGrad").Device(DEVICE_GPU), FemSourceGradOpGPU);

#endif