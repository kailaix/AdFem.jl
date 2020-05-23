#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "CholeskyOp.h"

REGISTER_OP("CholeskyLogdet")

.Input("a : double")
.Output("l : double")
.Output("jac : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));

        c->set_output(0, c->Matrix(c->Dim(c->input(0), 0),6));
        c->set_output(1, c->Vector(c->Dim(c->input(0), 0)));
    return Status::OK();
  });

class CholeskyLogdetOp : public OpKernel {
private:
  
public:
  explicit CholeskyLogdetOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape l_shape({n,6});
    TensorShape jac_shape({n});
            
    // create output tensor
    
    Tensor* l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, l_shape, &l));
    Tensor* jac = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jac_shape, &jac));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto l_tensor = l->flat<double>().data();
    auto jac_tensor = jac->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    factorize_logdet(jac_tensor, l_tensor, a_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("CholeskyLogdet").Device(DEVICE_CPU), CholeskyLogdetOp);





REGISTER_OP("CholeskyForwardOp")

.Input("a : double")
.Output("l : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));

        c->set_output(0, c->Matrix(c->Dim(c->input(0), 0),6));
    return Status::OK();
  });

REGISTER_OP("CholeskyForwardOpGrad")

.Input("grad_l : double")
.Input("l : double")
.Input("a : double")
.Output("grad_a : double");




REGISTER_OP("CholeskyBackwardOp")

.Input("l : double")
.Output("a : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle l_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &l_shape));

         c->set_output(0, c->Matrix(c->Dim(c->input(0), 0),9));
    return Status::OK();
  });

REGISTER_OP("CholeskyBackwardOpGrad")

.Input("grad_a : double")
.Input("a : double")
.Input("l : double")
.Output("grad_l : double");


class CholeskyForwardOpOp : public OpKernel {
private:
  
public:
  explicit CholeskyForwardOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape l_shape({n,6});
            
    // create output tensor
    
    Tensor* l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, l_shape, &l));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto l_tensor = l->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    factorize_forward(l_tensor, a_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("CholeskyForwardOp").Device(DEVICE_CPU), CholeskyForwardOpOp);



class CholeskyForwardOpGradOp : public OpKernel {
private:
  
public:
  explicit CholeskyForwardOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_l = context->input(0);
    const Tensor& l = context->input(1);
    const Tensor& a = context->input(2);
    
    
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(grad_l_shape.dims(), 2);
    DCHECK_EQ(l_shape.dims(), 2);
    DCHECK_EQ(a_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto l_tensor = l.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = a_shape.dim_size(0);
    factorize_backward(grad_a_tensor,grad_l_tensor, l_tensor, a_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("CholeskyForwardOpGrad").Device(DEVICE_CPU), CholeskyForwardOpGradOp);



class CholeskyBackwardOpOp : public OpKernel {
private:
  
public:
  explicit CholeskyBackwardOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& l = context->input(0);
    
    
    const TensorShape& l_shape = l.shape();
    
    
    DCHECK_EQ(l_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = l_shape.dim_size(0);
    TensorShape a_shape({n,9});
            
    // create output tensor
    
    Tensor* a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a_shape, &a));
    
    // get the corresponding Eigen tensors for data access
    
    auto l_tensor = l.flat<double>().data();
    auto a_tensor = a->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    outerproduct_forward(a_tensor, l_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("CholeskyBackwardOp").Device(DEVICE_CPU), CholeskyBackwardOpOp);



class CholeskyBackwardOpGradOp : public OpKernel {
private:
  
public:
  explicit CholeskyBackwardOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_a = context->input(0);
    const Tensor& a = context->input(1);
    const Tensor& l = context->input(2);
    
    
    const TensorShape& grad_a_shape = grad_a.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& l_shape = l.shape();
    
    
    DCHECK_EQ(grad_a_shape.dims(), 2);
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(l_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_l_shape(l_shape);
            
    // create output tensor
    
    Tensor* grad_l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_l_shape, &grad_l));
    
    // get the corresponding Eigen tensors for data access
    
    auto l_tensor = l.flat<double>().data();
    auto grad_a_tensor = grad_a.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto grad_l_tensor = grad_l->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = l_shape.dim_size(0);
    outerproduct_backward(grad_l_tensor, grad_a_tensor, a_tensor, l_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("CholeskyBackwardOpGrad").Device(DEVICE_CPU), CholeskyBackwardOpGradOp);

