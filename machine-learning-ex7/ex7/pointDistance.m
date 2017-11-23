function distance = pointDistance(point1, point2)
  distance = sqrt(sum((point1 - point2) .^ 2, 2));
end