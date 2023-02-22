import os 
import sys
import matplotlib.pyplot as plt
colors = []
for i in range(24):
    if i < 8:
        r = 0
        g = int(i/8 * 255)
        b = int((8 - i)/8 * 255)
    elif i < 16:
        r = int((i - 8)/8 * 255)
        g = int((16 - i)/8 * 255)
        b = int((i - 8)/8 * 255)
    else:
        r = int((24 - i)/8 * 255)
        g = 0
        b = int((i - 16)/8 * 255)
    color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    colors.append(color)
 
def plot_rank(lists):
    """
    Generates a line plot and a bar plot of the temperature lists for 24 cities.
    """
    # Set x-axis values as days (replace with your own x-axis labels)
    iterations = [x*200 for x in range(1, 41)]
    # Set line colors and markers (change as desired)
    
    # Determine the rank of each city for each day
    ranks = []
    for i in range(40):
        max_temps = [temps[i] for temps in lists]
        rank = sorted(range(len(max_temps)), key=lambda k: max_temps[k], reverse=True)
        ranks.append(rank)
    
    # Plot the rank of each city for each day as a separate line
    plt.figure()
    for i in range(24):
        if i % 3 != 0:
            continue
        city_rank = [ranks[j].index(i) + 1 for j in range(40)]
        plt.plot(iterations, city_rank, color=colors[i], label=f'Layer {i+1}')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Rank')
    
    # Show the plots
    plt.show()


def generate_line_plot(lists):
    """
    Generates a line plot of 24 temperature lists for 24 cities.
    """
    # Set x-axis values as days (replace with your own x-axis labels)
    iterations = [ x*200 for x in range(1, 41)]
    print(iterations)
    # Set line colors and markers (change as desired)

   #colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#FFA500', '#800080', '#00FF00', '#FF00FF', '#008080',
              #'#000080', '#808080', '#FFD700', '#008000', '#ADD8E6', '#FFC0CB', '#00FFFF', '#800000',
              #'#FF69B4', '#FFFF00', '#800080', '#00BFFF']
    
    print(len(colors))
    # Plot each temperature list as a separate line
    for i, temps in enumerate(lists):
        if i % 3 != 0:
            continue
        plt.plot(iterations, temps, color=colors[i], label=f'Layer {i+1}')

    # Add legend and axis labels
    plt.legend()
    plt.xlabel('Training Iterations')
    plt.ylabel('Averaged Importance')

    # Show the plot
    plt.show()


def main():
    
    lists_of_data = [[] for _ in range(24)] 
    f = open("plots.log", 'r')
    for idx, line in enumerate(f):
        data = line[1:-2].split(',')
        data = [float(item) for item in data]
        print(data)
        lists_of_data[idx].extend(data)
    
    generate_line_plot(lists_of_data)
    #plot_rank(lists_of_data)
if __name__ == "__main__":
    main()
