#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "EvalStrainOnBoundaryEdge.h"


REGISTER_OP("EvalStrainOnBoundaryEdge")
.Input("u : double")
.Input("edge : int64")
.Output("sigma : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle edge_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edge_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("EvalStrainOnBoundaryEdgeGrad")
.Input("grad_sigma : double")
.Input("sigma : double")
.Input("u : double")
.Input("edge : int64")
.Output("grad_u : double")
.Output("grad_edge : int64");

/*-------------------------------------------------------------------------------------*/

class EvalStrainOnBoundaryEdgeOp : public OpKernel {
private:
  
public:
  explicit EvalStrainOnBoundaryEdgeOp(OpKernelConstruction* context) : OpKernel(context) {

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
    TensorShape sigma_shape({LineIntegralN * n_elem,3});
            
    // create output tensor
    
    Tensor* sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sigma_shape, &sigma));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto sigma_tensor = sigma->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    sigma->flat<double>().setZero();
    MFEM::EvalStrainOnBoundaryEdgeForward(
      sigma_tensor, u_tensor, edge_tensor, n_elem);

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnBoundaryEdge").Device(DEVICE_CPU), EvalStrainOnBoundaryEdgeOp);



class EvalStrainOnBoundaryEdgeGradOp : public OpKernel {
private:
  
public:
  explicit EvalStrainOnBoundaryEdgeGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sigma = context->input(0);
    const Tensor& sigma = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& edge = context->input(3);
    
    
    const TensorShape& grad_sigma_shape = grad_sigma.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& edge_shape = edge.shape();
    
    
    DCHECK_EQ(grad_sigma_shape.dims(), 2);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(edge_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_edge_shape(edge_shape);
            
    // create output tensor
    int n_elem = edge_shape.dim_size(0);
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_edge = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edge_shape, &grad_edge));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto grad_sigma_tensor = grad_sigma.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    MFEM::EvalStrainOnBoundaryEdgeBackward(
      grad_u_tensor, grad_sigma_tensor, sigma_tensor, u_tensor, edge_tensor, n_elem);
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnBoundaryEdgeGrad").Device(DEVICE_CPU), EvalStrainOnBoundaryEdgeGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("EvalStrainOnBoundaryEdgeGpu")
.Input("u : double")
.Input("edge : int64")
.Output("sigma : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle edge_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edge_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("EvalStrainOnBoundaryEdgeGpuGrad")
.Input("grad_sigma : double")
.Input("sigma : double")
.Input("u : double")
.Input("edge : int64")
.Output("grad_u : double")
.Output("grad_edge : int64");

class EvalStrainOnBoundaryEdgeOpGPU : public OpKernel {
private:
  
public:
  explicit EvalStrainOnBoundaryEdgeOpGPU(OpKernelConstruction* context) : OpKernel(context) {

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
    
    TensorShape sigma_shape({-1,3});
            
    // create output tensor
    
    Tensor* sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sigma_shape, &sigma));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto edge_tensor = edge.flat<int64>().data();
    auto sigma_tensor = sigma->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnBoundaryEdgeGpu").Device(DEVICE_GPU), EvalStrainOnBoundaryEdgeOpGPU);

class EvalStrainOnBoundaryEdgeGradOpGPU : public OpKernel {
private:
  
public:
  explicit EvalStrainOnBoundaryEdgeGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sigma = context->input(0);
    const Tensor& sigma = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& edge = context->input(3);
    
    
    const TensorShape& grad_sigma_shape = grad_sigma.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& edge_shape = edge.shape();
    
    
    DCHECK_EQ(grad_sigma_shape.dims(), 2);
    DCHECK_EQ(sigma_shape.dims(), 2);
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
    auto grad_sigma_tensor = grad_sigma.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnBoundaryEdgeGpuGrad").Device(DEVICE_GPU), EvalStrainOnBoundaryEdgeGradOpGPU);

#endif