#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// TODO:
// 1. attr
// 2. use type T
REGISTER_OP("UpdateRoi")
    .Attr("fillval: float = 1.0")
    .Input("w_mask: float")
    .Input("roi: float") // TODO: better if int
    .Output("w_mask_updated: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class UpdateRoiOp : public OpKernel {
    public:
        explicit UpdateRoiOp(OpKernelConstruction* context) : OpKernel(context) {
            // get attrivbutes and run checks
            OP_REQUIRES_OK(context, context->GetAttr("fillval", &fillval_));
            OP_REQUIRES(context, fillval_ >= 0.0, 
                errors::InvalidArgument("Need fillval >= 0.0, got ", fillval_));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor (w_mask)
            const Tensor& w_mask_tensor = context->input(0);
            OP_REQUIRES(context, w_mask_tensor.dims()==4, 
                    errors::InvalidArgument("w_mask tensor should have the dimension of 4"))
            auto w_mask = w_mask_tensor.tensor<float, 4>();

            int b = w_mask_tensor.dim_size(0);
            int h = w_mask_tensor.dim_size(1); // TODO: is it width or height?
            int w = w_mask_tensor.dim_size(2); // TODO: is it width or height?
            int c = w_mask_tensor.dim_size(3);
            //printf("w_mask_tensor dim size: %d %d %d %d \n", b, h, w, c);

            // Grab the input tensor (roi. ie, y_prev)
            const Tensor& roi_tensor = context->input(1);
            OP_REQUIRES(context, roi_tensor.dims()==2, 
                    errors::InvalidArgument("roi tensor should have the dimension of 2"));
            auto roi = roi_tensor.tensor<float, 2>();

            // check roi
            for (int bi=0; bi<b; bi++) {
                //OP_REQUIRES(context, roi(bi).NumElements()==4, 
                    //errors::InvalidArgument("roi tensor should have 4 elements"));
                int x1 = roi(bi, 0);
                int y1 = roi(bi, 1);
                int x2 = roi(bi, 2);
                int y2 = roi(bi, 3);
                //printf("(x1,y1,x2,y2) = (%d, %d, %d, %d) \n", x1, y1, x2, y2);
                OP_REQUIRES(context, x2>=x1, errors::InvalidArgument("x2 should be >= x1"));
                OP_REQUIRES(context, y2>=y1, errors::InvalidArgument("y2 should be >= y1"));
                OP_REQUIRES(context, x2<w, errors::InvalidArgument("x2 should be < width"));
                OP_REQUIRES(context, y2<h, errors::InvalidArgument("y2 should be < height"));
            }

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, w_mask_tensor.shape(), &output_tensor));
            auto output = output_tensor->tensor<float, 4>();

            // TODO: there should be a better way of copying from w_mask
            for (int bi=0; bi<b; bi++) {
                for (int hi=0; hi<h; hi++) {
                    for (int wi=0; wi<w; wi++) {
                        for (int ci=0; ci<c; ci++) {
                            output(bi,hi,wi,ci) = w_mask(bi,hi,wi,ci);
                        }
                    }
                }
            }

            // main operation
            // TODO: try different types of weighting instead of box
            for (int bi=0; bi<b; bi++) {
                int x1 = roi(bi, 0);
                int y1 = roi(bi, 1);
                int x2 = roi(bi, 2);
                int y2 = roi(bi, 3);
                for (int hi=y1; hi<=y2; hi++) {
                    for (int wi=x1; wi<=x2; wi++) {
                        for (int ci=0; ci<c; ci++) {
                            output(bi,hi,wi,ci) = fillval_;
                        }
                    }
                }
            }
        }

    private:
        float fillval_;
};

REGISTER_KERNEL_BUILDER(Name("UpdateRoi").Device(DEVICE_CPU), UpdateRoiOp);
//REGISTER_KERNEL_BUILDER(Name("UpdateRoi").Device(DEVICE_GPU), UpdateRoiOp);
// TODO: separate GPU register? type register


