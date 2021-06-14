#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "EvalScalarOnBoundaryEdge.h"


REGISTER_OP("EvalScalarOnBoundaryEdge")
.Input("u : double")
.Input("edge : int64")
.Output("s : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle edge_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edge_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("EvalScalarOnBoundaryEdgeGrad")
.Input("grad_s : double")
.Input("s : double")
.Input("u : double")
.Input("edge : int64")
.Output("grad_u : double")
.Output("grad_edge : int64");

/*-------------------------------------------------------------------------------------*/

class EvalScalarOnBoundaryEdgeOp : public OpKernel {
private:
  
public:
  explicit EvalScalarOnBoundaryEdgeOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& edge = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& edge_shape = edge.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(edge_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n_elem = edge_shape.dim_size(0);
    int LineIntegralN = get_LineIntegralN();
    TensorShape s_shape({LineIntegralN*n_elem});
            
    // create output tensor
    
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, s_shape, &s));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto s_tensor = s->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::EvalScalarOnBoundaryEdgeForward(s_tensor, u_tensor, 
      edge_tensor, n_elem);

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalScalarOnBoundaryEdge").Device(DEVICE_CPU), EvalScalarOnBoundaryEdgeOp);



class EvalScalarOnBoundaryEdgeGradOp : public OpKernel {
private:
  
public:
  explicit EvalScalarOnBoundaryEdgeGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_s = context->input(0);
    const Tensor& s = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& edge = context->input(3);
    
    
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& edge_shape = edge.shape();
    
    
    DCHECK_EQ(grad_s_shape.dims(), 1);
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(edge_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
    int n_elem = edge_shape.dim_size(0);
    
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_edge_shape(edge_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_edge = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edge_shape, &grad_edge));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    MFEM::EvalScalarOnBoundaryEdgeBackward(
      grad_u_tensor, grad_s_tensor, s_tensor, u_tensor, edge_tensor, n_elem);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalScalarOnBoundaryEdgeGrad").Device(DEVICE_CPU), EvalScalarOnBoundaryEdgeGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("EvalScalarOnBoundaryEdgeGpu")
.Input("u : double")
.Input("edge : int64")
.Output("s : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle edge_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edge_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("EvalScalarOnBoundaryEdgeGpuGrad")
.Input("grad_s : double")
.Input("s : double")
.Input("u : double")
.Input("edge : int64")
.Output("grad_u : double")
.Output("grad_edge : int64");

class EvalScalarOnBoundaryEdgeOpGPU : public OpKernel {
private:
  
public:
  explicit EvalScalarOnBoundaryEdgeOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& edge = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& edge_shape = edge.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(edge_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape s_shape({-1});
            
    // create output tensor
    
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, s_shape, &s));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto s_tensor = s->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalScalarOnBoundaryEdgeGpu").Device(DEVICE_GPU), EvalScalarOnBoundaryEdgeOpGPU);

class EvalScalarOnBoundaryEdgeGradOpGPU : public OpKernel {
private:
  
public:
  explicit EvalScalarOnBoundaryEdgeGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_s = context->input(0);
    const Tensor& s = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& edge = context->input(3);
    
    
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& edge_shape = edge.shape();
    
    
    DCHECK_EQ(grad_s_shape.dims(), 1);
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(edge_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_edge_shape(edge_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_edge = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edge_shape, &grad_edge));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalScalarOnBoundaryEdgeGpuGrad").Device(DEVICE_GPU), EvalScalarOnBoundaryEdgeGradOpGPU);

#endif