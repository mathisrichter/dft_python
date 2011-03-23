import DynamicField
import GaussKernel

def main():
    interaction_kernel = GaussKernel.GaussKernel(1)
    interaction_kernel.add_mode(1.0, [0.5], [0.0])
    interaction_kernel.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel.calculate()

    field = DynamicField.DynamicField([3,4,5], None)

    projection = DynamicField.Projection(3, 0, set([]), [])

    projection.set_input_dimensionality(3)
    projection.set_output_dimensionality(0)
    projection.set_input_dimension_sizes([3,4,5])
    projection.set_output_dimension_sizes([1])
    projection._incoming_connectables.append(field)

#    for i in range(0, 500):
    field.step()
    projection.step()
    output = projection.get_output()
    print(output)
    print(output.shape)

if __name__ == "__main__":
    main()
