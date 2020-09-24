#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ImposeDirichlet.h"


REGISTER_OP("ImposeDirichlet")
.Input("indices : int64")
.Input("vv : double")
.Input("bd : int64")
.Input("rhs : double")
.Input("bdval : double")
.Output("oindices : int64")
.Output("ov : double")
.Output("orhs : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vv_shape));
        shape_inference::ShapeHandle bd_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bd_shape));
        shape_inference::ShapeHandle rhs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &rhs_shape));
        shape_inference::ShapeHandle bdval_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &bdval_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ImposeDirichletGrad")
.Input("grad_ov : double")
.Input("grad_orhs : double")
.Input("oindices : int64")
.Input("ov : double")
.Input("orhs : double")
.Input("indices : int64")
.Input("vv : double")
.Input("bd : int64")
.Input("rhs : double")
.Input("bdval : double")
.Output("grad_indices : int64")
.Output("grad_vv : double")
.Output("grad_bd : int64")
.Output("grad_rhs : double")
.Output("grad_bdval : double");

/*-------------------------------------------------------------------------------------*/

class ImposeDirichletOp : public OpKernel {
private:
  
public:
  explicit ImposeDirichletOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& indices = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& bd = context->input(2);
    const Tensor& rhs = context->input(3);
    const Tensor& bdval = context->input(4);
    
    
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& bdval_shape = bdval.shape();
    
    
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(bdval_shape.dims(), 1);

    // extra check
        
    // create output shape
    int sN = indices_shape.dim_size(0);
    int N = rhs_shape.dim_size(0);
    int bdN = bd_shape.dim_size(0);
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int64>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto bdval_tensor = bdval.flat<double>().data();
    MFEM::ImposeDirichlet ID;
    ID.ImposeDirichlet_forward(indices_tensor, vv_tensor, bd_tensor, rhs_tensor, bdval_tensor, N, bdN, sN);
    int Size = ID.vv.size();
    
    TensorShape oindices_shape({Size,2});
    TensorShape ov_shape({Size});
    TensorShape orhs_shape({N});
            
    // create output tensor
    
    Tensor* oindices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, oindices_shape, &oindices));
    Tensor* ov = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, ov_shape, &ov));
    Tensor* orhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, orhs_shape, &orhs));
    
    // get the corresponding Eigen tensors for data access
    
    
    auto oindices_tensor = oindices->flat<int64>().data();
    auto ov_tensor = ov->flat<double>().data();
    auto orhs_tensor = orhs->flat<double>().data();   

    // implement your forward function here 
    ID.Copy(oindices_tensor, ov_tensor, orhs_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ImposeDirichlet").Device(DEVICE_CPU), ImposeDirichletOp);



class ImposeDirichletGradOp : public OpKernel {
private:
  
public:
  explicit ImposeDirichletGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ov = context->input(0);
    const Tensor& grad_orhs = context->input(1);
    const Tensor& oindices = context->input(2);
    const Tensor& ov = context->input(3);
    const Tensor& orhs = context->input(4);
    const Tensor& indices = context->input(5);
    const Tensor& vv = context->input(6);
    const Tensor& bd = context->input(7);
    const Tensor& rhs = context->input(8);
    const Tensor& bdval = context->input(9);
    
    
    const TensorShape& grad_ov_shape = grad_ov.shape();
    const TensorShape& grad_orhs_shape = grad_orhs.shape();
    const TensorShape& oindices_shape = oindices.shape();
    const TensorShape& ov_shape = ov.shape();
    const TensorShape& orhs_shape = orhs.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& bdval_shape = bdval.shape();
    
    
    DCHECK_EQ(grad_ov_shape.dims(), 1);
    DCHECK_EQ(grad_orhs_shape.dims(), 1);
    DCHECK_EQ(oindices_shape.dims(), 2);
    DCHECK_EQ(ov_shape.dims(), 1);
    DCHECK_EQ(orhs_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(bdval_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_indices_shape(indices_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_bd_shape(bd_shape);
    TensorShape grad_rhs_shape(rhs_shape);
    TensorShape grad_bdval_shape(bdval_shape);
            
    // create output tensor
    
    Tensor* grad_indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_indices_shape, &grad_indices));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_bd = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_bd_shape, &grad_bd));
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_rhs_shape, &grad_rhs));
    Tensor* grad_bdval = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_bdval_shape, &grad_bdval));
    
    // get the corresponding Eigen tensors for data access
    
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int64>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto bdval_tensor = bdval.flat<double>().data();
    auto grad_ov_tensor = grad_ov.flat<double>().data();
    auto grad_orhs_tensor = grad_orhs.flat<double>().data();
    auto oindices_tensor = oindices.flat<int64>().data();
    auto ov_tensor = ov.flat<double>().data();
    auto orhs_tensor = orhs.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();
    auto grad_bdval_tensor = grad_bdval->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int sN = indices_shape.dim_size(0);
    int N = rhs_shape.dim_size(0);
    int bdN = bd_shape.dim_size(0);
    MFEM::ImposeDirichlet ID;
    grad_vv->flat<double>().setZero();
    grad_rhs->flat<double>().setZero();
    grad_bdval->flat<double>().setZero();
    ID.ImposeDirichlet_backward(
        grad_vv_tensor, grad_rhs_tensor, grad_bdval_tensor, 
        grad_ov_tensor, grad_orhs_tensor, 
        indices_tensor, vv_tensor, bd_tensor, rhs_tensor, bdval_tensor, N, bdN, sN);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ImposeDirichletGrad").Device(DEVICE_CPU), ImposeDirichletGradOp);
