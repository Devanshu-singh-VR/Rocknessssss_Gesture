def FirstFit(block_Size, m, process_Size, n):


# code to store the block id of the block that needs to be allocated to a process
    allocate = [-1] * n

# Any process is assigned with the memory at the initial stage

# find a suitable block for each process
# the blocks are allocated as per their size


    for i in range(n):
        for j in range(m):
            if block_Size[j] >= process_Size[i]:
                # assign the block j to p[i] process
                allocate[i] = j

                # available block memory is reduced
                block_Size[j] -= process_Size[i]

                break

            print(" Process Number Process Size Block Number")

            for i in range(n):

                print(" ", i + 1, "      ", process_Size[i], " ", end=" ")

            if allocate[i] != -1:
                print(allocate[i] + 1)
    else:

        print("Not Allocated")

    # Driver code

if __name__ == '__main__':
    b_Size = [50, 200, 70, 115, 15]
    p_Size = [100, 10, 35, 15, 23, 6, 25]
    m = len(b_Size)
    n = len(p_Size)

FirstFit(b_Size, m, p_Size, n)