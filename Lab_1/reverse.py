def reverse(equation: str):
    if len(equation) < 1:
        return ""
    else:
        token = " "

        for i in equation[::-1]:
            if i != " ":
                token += i

            else:
                token += ""
                break

        b = equation[::-1].strip(token)[::-1]
        token = token[::-1]
        print(b)
    return token + reverse(b)


print(reverse("70 - 39 + 15 * 26"))
