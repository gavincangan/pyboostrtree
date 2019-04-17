#include "RTreePoint3D.hpp"
#include <iostream>
#include <cassert>

using namespace rtrees;

RTreePoint3D::RTreePoint3D() {
    // std::cout << "RTreePoint3D() init" << std::endl;
}

RTreePoint3D::~RTreePoint3D() {
        // std::cout << "RTreePoint3D() destruct" << std::endl;
}

void RTreePoint3D::insertPoint(double x, double y, double z)
{
    // std::cout << "insertPoint: (" << x <<", "<< y <<", "<< z << ")" << std::endl;
    point3d p(x, y, z);
    this->rtree.insert(std::make_pair(p, this->points.size()));

    std::vector<double> point{x, y, z};
    this->points.push_back(point);
}

void RTreePoint3D::insertPoints(double* points, long m, long n)
{
    // for(long j=0; j<m; j++)
        // std::cout << "insertPoints: (" << points[j+0] <<", "<< points[j+1] <<", "<< points[j+2] << ") -- " << j << "/" << m << " " << n << std::endl;
    assert(n == 3); // points should be an m x 3 matrix
    std::vector<double> point(3);
    for (long i = 0 ; i < m * n; i = i + 3){
        point3d p(points[i], points[i + 1], points[i + 2]);
        this->rtree.insert(std::make_pair(p, this->points.size()));

        point[0] = points[i]; point[1] = points[i+1]; point[2] = points[i+2];
        this->points.push_back(point);
    }
}

long RTreePoint3D::size(){
    // std::cout << "_all_points" << std::endl;
    return this->rtree.size();
}

std::vector<long> RTreePoint3D::knn(double x, double y, double z, int k){
    // std::cout << "knn: (" << x <<", "<< y <<", "<< z << ") -- " << k << std::endl;
    point3d p(x, y, z);
    std::vector<value3d> results;
    rtree.query(bgi::nearest(p, k), std::back_inserter(results));
    std::vector<long> values;
    for (auto result : results){
        values.insert(values.begin(), result.second);
    }
    return values;
}

std::vector<long> RTreePoint3D::knn_np(double* coords, int k){
    // std::cout << "knn_np: (" << coords[0] <<", "<< coords[1] <<", "<< coords[2] << ") -- " << k << std::endl;
    point3d p(coords[0], coords[1], coords[2]);
    std::vector<value3d> results;
    // std::cout << "Sending query to bgi::nearest" << std::endl;
    rtree.query(bgi::nearest(p, k), std::back_inserter(results));
    // std::cout << "Results from bgi::nearest -- ";
    std::vector<long> values;
    for (auto result : results){
        // std::cout << result.second << ",  ";
        values.insert(values.begin(), result.second);
    }
    // std::cout << std::endl;
    return values;
}


double RTreePoint3D::minDistance(double x, double y, double z){
    // std::cout << "minDistance: (" << x <<", "<< y <<", "<< z << ")" << std::endl;
    point3d p(x, y, z);
    std::vector<value3d> results;
    rtree.query(bgi::nearest(p, 1), std::back_inserter(results));

    double dist = std::numeric_limits<double>::max();
	for (const value3d& v: results){
		double localDist = bg::distance(p, v.first);
		if (localDist < dist) dist = localDist;
	}

    return dist;
}

std::vector<double> RTreePoint3D::bounds(){
    // std::cout << "bounds" << std::endl;
    auto bbox = this->rtree.bounds();
    auto min_corner = bbox.min_corner();
    auto max_corner = bbox.max_corner();

    std::vector<double> boundaries;
    boundaries.push_back(bg::get<0>(min_corner));
    boundaries.push_back(bg::get<1>(min_corner));
    boundaries.push_back(bg::get<2>(min_corner));
    boundaries.push_back(bg::get<0>(max_corner));
    boundaries.push_back(bg::get<1>(max_corner));
    boundaries.push_back(bg::get<2>(max_corner));

    return boundaries;
}

std::vector<long> RTreePoint3D::intersection(double* coords){
    // std::cout << "intersection: (" << coords[0] <<", "<< coords[1] <<", "<< coords[2] <<")\t("<< coords[3] <<", "<< coords[4] <<", "<< coords[5] << ")" << std::endl;
    _sortMinMaxCorners(coords, 2, 3);
    point3d minp(coords[0], coords[1], coords[2]), maxp(coords[3], coords[4], coords[5]);
    bbox3d query_box(minp, maxp);

    std::vector<value3d> results;
    rtree.query(bgi::intersects(query_box), std::back_inserter(results));
    std::vector<long> values;
    for (auto result : results){
        values.insert(values.begin(), result.second);
    }
    return values;
}

void RTreePoint3D::removePoints(double* points, long m, long n){
    // std::cout << "removePoints: (" << points[0] <<", "<< points[1] <<", "<< points[2] << ") -- " << m << " " << n << std::endl;
    assert(n == 4); // points should be an m x 3 matrix

    for (long i = 0 ; i < m * n; i = i + n){
        point3d p(points[i], points[i + 1], points[i + 2]);
        auto nearest = knn(points[i], points[i + 1], points[i + 2], 1);
        this->rtree.remove(std::make_pair(p, nearest[0]));
    }
}

void RTreePoint3D::_sortMinMaxCorners(double* points, long m, long n)
{
    // std::cout << "_sortMinMaxCorners: (" << points[0] <<", "<< points[1] <<", "<< points[2] <<")\t("<< points[3] <<", "<< points[4] <<", "<< points[5] << ") -- " << m << " " << n << std::endl;
    assert(n == 3);
    assert(m == 2);

    double temp = 0.0;
    for (long i = 0 ; i < n; i++){
        if(points[i] > points[i+n]){
            temp = points[i];
            points[i] = points[i+n];
            points[i+n] = temp;
        }
    }
}

std::vector<std::vector<double>> RTreePoint3D::_all_points(){
    // std::cout << "_all_points" << std::endl;
    return this->points;
}