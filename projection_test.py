import DynamicField
import Kernel

def main():
    interaction_kernel_0 = Kernel.GaussKernel(3)
    interaction_kernel_0.add_mode(1.0, [0.5,0.5,0.5], [0.0,0.0,0.0])
    interaction_kernel_0.add_mode(-5.5, [5.5,5.5,5.5], [0.0,0.0,0.0])
    interaction_kernel_0.calculate()

    interaction_kernel_1 = Kernel.GaussKernel(1)
    interaction_kernel_1.add_mode(1.0, [0.5], [0.0])
    interaction_kernel_1.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel_1.calculate()

    field_0 = DynamicField.DynamicField([[5],[3],[4]], [], interaction_kernel_0)
    field_0.set_name("Field0")
    field_0.set_boost(20)
    field_1 = DynamicField.DynamicField([[5]], [], interaction_kernel_1)
    field_1.set_name("Field1")


    scaler = DynamicField.Scaler()
    scaler.set_name("Scaler")

    weight = DynamicField.Weight([0., 0.5, 1.0, 0.5, 0.0])
    weight.set_name("Weight")

    projection_0 = DynamicField.Projection(3, 1, set([1]), [0])
    projection_0.set_name("Projection0")

    processing_steps = [projection_0, scaler, weight]
    
    DynamicField.connect(field_0, field_1, processing_steps)

    #projection.set_input_dimensionality(3)
    #projection.set_output_dimensionality(0)
    #projection.set_input_dimension_sizes([3,4,5])
    #projection.set_output_dimension_sizes([1])
    #projection._incoming_connectables.append(field)

    for i in range(0, 500):
        field_0.step()
        print("field 0:")
        print(field_0.get_output())
        print(field_0.get_output().shape)
        for processing_step in processing_steps:
            processing_step.step()
            print(processing_step.get_name() + ": ")
            print(processing_step.get_output())
            print(processing_step.get_output().shape)
        field_1.step()
        print("field 1:")
        print(field_1.get_activation())
        print(field_1.get_activation().shape)

if __name__ == "__main__":
    main()
