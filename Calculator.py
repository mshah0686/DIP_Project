def calculator(nums):
    cleaned = []
    current = ""
    for num in nums:
        if num is not "+" and num is not "-" and num is not "*" and num is not "/":
            current = current + num
        else:
            cleaned.append(current)
            cleaned.append(num)
            current = ""
    cleaned.append(current)
    final = ""
    for element in cleaned:
        final = final + element
    code_str = "x="+(final)+"\nprint(x)"
    code = compile(code_str, "sum.py", "exec")
    return code


if __name__ == '__main__':
    data = ["1", "0", "-", "2", "*", "3"]
    result = calculator(data)
    eval(result)    

