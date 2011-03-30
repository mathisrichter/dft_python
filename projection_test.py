import DynamicField
import Kernel

def main():
    interaction_kernel = Kernel.GaussKernel(1)
    interaction_kernel.add_mode(1.0, [0.5], [0.0])
    interaction_kernel.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel.calculate()

    field_0 = DynamicField.DynamicField([[10],[4]], [], None)
    field_0.set_name("Field0")
    field_1 = DynamicField.DynamicField([[3],[5]], [], None)
    field_1.set_name("Field1")

    scaler = DynamicField.Scaler()
    scaler.set_name("Scaler")

    projection_0 = DynamicField.Projection(2, 3, set([0,1]), [1,0])
    projection_0.set_name("Projection0")

    processing_steps = [scaler]
    
    DynamicField.connect(field_0, field_1, processing_steps)

    #projection.set_input_dimensionality(3)
    #projection.set_output_dimensionality(0)
    #projection.set_input_dimension_sizes([3,4,5])
    #projection.set_output_dimension_sizes([1])
    #projection._incoming_connectables.append(field)

#    for i in range(0, 500):
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
    print(field_1.get_output())
    print(field_1.get_output().shape)

if __name__ == "__main__":
    main()
