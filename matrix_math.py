class ShapeException(Exception):
    pass


def shape_vectors(matrix):
    """shape should take a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    # length = len(vector)
    # return (length, )
    if isinstance(matrix[0], int):
        return (len(matrix), )

    else:
        return (len(matrix[0]), len(matrix))
#print(shape_vectors(q))


def vector_add(vector1, vector2):
    """
    [a b]  + [c d]  = [a+c b+d]

    Matrix + Matrix = Matrix
    """
    # list = []
    # for i in range(len(vector1))
    #  return list.append(vector1[i] + vector2[i])

    return [vector1[i] + vector2[i] for i in range(len(vector1))]


def vector_sub(vector1, vector2):
    """
    [a b]  - [c d]  = [a-c b-d]

    Matrix + Matrix = Matrix
    """

    # list = []
    # for i in range(len(vector1))
    #  return list.append(vector1[i] - vector2[i])

    return [vector1[i] - vector2[i] for i in range(len(vector1))]


def vector_sum(*vectors):
    """vector_sum can take any number of vectors and add them together."""

    sum = vectors[0]
    for vec in vectors[1:]:
        sum = vector_add(sum, vec)
    return sum
#recursive version
    # if len(vector) = 1:
    #     return vector[0]
    # else:
    #     return vector_add(vector[0] + vector_sum(vector[1:])

def dot(vector1, vector2):
    """
    dot([a b], [c d])   = a * c + b * d

    dot(Vector, Vector) = Scalar
    """
    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])


def vector_multiply(vector, scalor):
    """
    [a b]  *  Z     = [a*Z b*Z]

    Vector * Scalar = Vector
    """

    return [vector[i] * scalor for i in range(len(vector))]


def vector_mean(*vectors):
    """
    mean([a b], [c d]) = [mean(a, c) mean(b, d)]

    mean(Vector)       = Vector
    """

    return [vector_sum(*vectors)[i] / len(vectors) for i in range(len(vector_sum(*vectors)))]


def magnitude(vector):
    """
    magnitude([a b])  = sqrt(a^2 + b^2)

    magnitude(Vector) = Scalar
    """
    assert magnitude(m) == 5
    assert magnitude(v) == math.sqrt(10)
    assert magnitude(y) == math.sqrt(1400)
    assert magnitude(z) == 0

    return [ for vector[i]]
