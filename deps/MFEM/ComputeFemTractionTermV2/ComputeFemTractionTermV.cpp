#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeFemTractionTermV.h"


REGISTER_OP("ComputeFemTractionTermV")
.Input("t : double")
.Input("edgeid : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &t_shape));
        shape_inference::ShapeHandle edgeid_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edgeid_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemTractionTermVGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("t : double")
.Input("edgeid : int64")
.Output("grad_t : double")
.Output("grad_edgeid : int64");

/*-------------------------------------------------------------------------------------*/

class ComputeFemTractionTermVOp : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermVOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& t = context->input(0);
    const Tensor& edgeid = context->input(1);
    
    
    const TensorShape& t_shape = t.shape();
    const TensorShape& edgeid_shape = edgeid.shape();
    
    
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(edgeid_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = edgeid_shape.dim_size(0);
    TensorShape out_shape({mmesh.ndof});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto edgeid_tensor = edgeid.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    MFEM::ComputeFemTractionV_forward(out_tensor, 
      t_tensor, edgeid_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermV").Device(DEVICE_CPU), ComputeFemTractionTermVOp);



class ComputeFemTractionTermVGradOp : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermVGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& t = context->input(2);
    const Tensor& edgeid = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& edgeid_shape = edgeid.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(edgeid_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_t_shape(t_shape);
    TensorShape grad_edgeid_shape(edgeid_shape);
            
    // create output tensor
    
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_t_shape, &grad_t));
    Tensor* grad_edgeid = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edgeid_shape, &grad_edgeid));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto edgeid_tensor = edgeid.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = edgeid_shape.dim_size(0);
    MFEM::ComputeFemTraction_backward(
         grad_t_tensor, 
        grad_out_tensor, 
        out_tensor, t_tensor, 
        edgeid_tensor, n
    );
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermVGrad").Device(DEVICE_CPU), ComputeFemTractionTermVGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("ComputeFemTractionTermVGpu")
.Input("t : double")
.Input("edgeid : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &t_shape));
        shape_inference::ShapeHandle edgeid_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edgeid_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemTractionTermVGpuGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("t : double")
.Input("edgeid : int64")
.Output("grad_t : double")
.Output("grad_edgeid : int64");

class ComputeFemTractionTermVOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermVOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& t = context->input(0);
    const Tensor& edgeid = context->input(1);
    
    
    const TensorShape& t_shape = t.shape();
    const TensorShape& edgeid_shape = edgeid.shape();
    
    
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(edgeid_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto edgeid_tensor = edgeid.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermVGpu").Device(DEVICE_GPU), ComputeFemTractionTermVOpGPU);

class ComputeFemTractionTermVGradOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermVGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& t = context->input(2);
    const Tensor& edgeid = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& edgeid_shape = edgeid.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(edgeid_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_t_shape(t_shape);
    TensorShape grad_edgeid_shape(edgeid_shape);
            
    // create output tensor
    
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_t_shape, &grad_t));
    Tensor* grad_edgeid = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_edgeid_shape, &grad_edgeid));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto edgeid_tensor = edgeid.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermVGpuGrad").Device(DEVICE_GPU), ComputeFemTractionTermVGradOpGPU);

#endif