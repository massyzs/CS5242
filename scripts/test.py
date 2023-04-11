from random import randint
commands=[randint(1,10) for i in range(10) ]
max_parallel_commands=4
command_groups = [commands[i:i + max_parallel_commands] for i in range(0, len(commands), max_parallel_commands)]
print(command_groups)