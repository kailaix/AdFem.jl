#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemToGaussPoints.h"


REGISTER_OP("FemToGaussPoints")
.Input("u : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("ugauss : double")
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

REGISTER_OP("FemToGaussPointsGrad")
.Input("grad_ugauss : double")
.Input("ugauss : double")
.Input("u : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class FemToGaussPointsOp : public OpKernel {
private:
  
public:
  explicit FemToGaussPointsOp(OpKernelConstruction* context) : OpKernel(context) {

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
    
    TensorShape ugauss_shape({4*m_*n_});
            
    // create output tensor
    
    Tensor* ugauss = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ugauss_shape, &ugauss));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    
    auto h_tensor = h.flat<double>().data();
    auto ugauss_tensor = ugauss->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    FemToGaussPoints_forward(ugauss_tensor, u_tensor, *m_tensor, *n_tensor, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemToGaussPoints").Device(DEVICE_CPU), FemToGaussPointsOp);



class FemToGaussPointsGradOp : public OpKernel {
private:
  
public:
  explicit FemToGaussPointsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ugauss = context->input(0);
    const Tensor& ugauss = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_ugauss_shape = grad_ugauss.shape();
    const TensorShape& ugauss_shape = ugauss.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_ugauss_shape.dims(), 1);
    DCHECK_EQ(ugauss_shape.dims(), 1);
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
    auto grad_ugauss_tensor = grad_ugauss.flat<double>().data();
    auto ugauss_tensor = ugauss.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    FemToGaussPoints_backward(grad_u_tensor, grad_ugauss_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemToGaussPointsGrad").Device(DEVICE_CPU), FemToGaussPointsGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class FemToGaussPointsOpGPU : public OpKernel {
private:
  
public:
  explicit FemToGaussPointsOpGPU(OpKernelConstruction* context) : OpKernel(context) {

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
    
    TensorShape ugauss_shape({-1});
            
    // create output tensor
    
    Tensor* ugauss = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ugauss_shape, &ugauss));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto ugauss_tensor = ugauss->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemToGaussPoints").Device(DEVICE_GPU), FemToGaussPointsOpGPU);

class FemToGaussPointsGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemToGaussPointsGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ugauss = context->input(0);
    const Tensor& ugauss = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_ugauss_shape = grad_ugauss.shape();
    const TensorShape& ugauss_shape = ugauss.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_ugauss_shape.dims(), 1);
    DCHECK_EQ(ugauss_shape.dims(), 1);
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
    auto grad_ugauss_tensor = grad_ugauss.flat<double>().data();
    auto ugauss_tensor = ugauss.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemToGaussPointsGrad").Device(DEVICE_GPU), FemToGaussPointsGradOpGPU);

#endif