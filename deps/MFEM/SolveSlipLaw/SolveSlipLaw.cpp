#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "SolveSlipLaw.h"


REGISTER_OP("SolveSlipLaw")
.Input("a : double")
.Input("b : double")
.Input("c : double")
.Input("xinit : double")
.Output("x : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &b_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &c_shape));
        shape_inference::ShapeHandle xinit_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &xinit_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("SolveSlipLawGrad")
.Input("grad_x : double")
.Input("x : double")
.Input("a : double")
.Input("b : double")
.Input("c : double")
.Input("xinit : double")
.Output("grad_a : double")
.Output("grad_b : double")
.Output("grad_c : double")
.Output("grad_xinit : double");

/*-------------------------------------------------------------------------------------*/

class SolveSlipLawOp : public OpKernel {
private:
  
public:
  explicit SolveSlipLawOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& xinit = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& xinit_shape = xinit.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(xinit_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape x_shape({n});
            
    // create output tensor
    
    Tensor* x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &x));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto xinit_tensor = xinit.flat<double>().data();
    auto x_tensor = x->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::SolveSlipLawForward(
      x_tensor, xinit_tensor, a_tensor, b_tensor, c_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveSlipLaw").Device(DEVICE_CPU), SolveSlipLawOp);



class SolveSlipLawGradOp : public OpKernel {
private:
  
public:
  explicit SolveSlipLawGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_x = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& b = context->input(3);
    const Tensor& c = context->input(4);
    const Tensor& xinit = context->input(5);
    
    
    const TensorShape& grad_x_shape = grad_x.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& xinit_shape = xinit.shape();
    
    
    DCHECK_EQ(grad_x_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(xinit_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int n = a_shape.dim_size(0);
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_xinit_shape(xinit_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_xinit = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_xinit_shape, &grad_xinit));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto xinit_tensor = xinit.flat<double>().data();
    auto grad_x_tensor = grad_x.flat<double>().data();
    auto x_tensor = x.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_xinit_tensor = grad_xinit->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    MFEM::SolveSlipLawBackward(
      grad_a_tensor, grad_b_tensor, grad_c_tensor, grad_x_tensor,
      x_tensor, a_tensor, b_tensor, c_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveSlipLawGrad").Device(DEVICE_CPU), SolveSlipLawGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("SolveSlipLawGpu")
.Input("a : double")
.Input("b : double")
.Input("c : double")
.Input("xinit : double")
.Output("x : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &b_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &c_shape));
        shape_inference::ShapeHandle xinit_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &xinit_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("SolveSlipLawGpuGrad")
.Input("grad_x : double")
.Input("x : double")
.Input("a : double")
.Input("b : double")
.Input("c : double")
.Input("xinit : double")
.Output("grad_a : double")
.Output("grad_b : double")
.Output("grad_c : double")
.Output("grad_xinit : double");

class SolveSlipLawOpGPU : public OpKernel {
private:
  
public:
  explicit SolveSlipLawOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& xinit = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& xinit_shape = xinit.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(xinit_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape x_shape({-1});
            
    // create output tensor
    
    Tensor* x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &x));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto xinit_tensor = xinit.flat<double>().data();
    auto x_tensor = x->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SolveSlipLawGpu").Device(DEVICE_GPU), SolveSlipLawOpGPU);

class SolveSlipLawGradOpGPU : public OpKernel {
private:
  
public:
  explicit SolveSlipLawGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_x = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& b = context->input(3);
    const Tensor& c = context->input(4);
    const Tensor& xinit = context->input(5);
    
    
    const TensorShape& grad_x_shape = grad_x.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& xinit_shape = xinit.shape();
    
    
    DCHECK_EQ(grad_x_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(xinit_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_xinit_shape(xinit_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_xinit = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_xinit_shape, &grad_xinit));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto xinit_tensor = xinit.flat<double>().data();
    auto grad_x_tensor = grad_x.flat<double>().data();
    auto x_tensor = x.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_xinit_tensor = grad_xinit->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveSlipLawGpuGrad").Device(DEVICE_GPU), SolveSlipLawGradOpGPU);

#endif