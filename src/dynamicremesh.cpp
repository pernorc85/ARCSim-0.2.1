/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.
  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.
  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

/*
 * Edited by Le Yang. 2019.
 * Changed the data structure for mesh to DCEL.
 */

#include "dynamicremesh.h"
#include "remesh.h"
#include "cloth_physics.h"
//#include "tensormax.hpp"
//#include "timer.hpp"
//#include "util.hpp"
#include <algorithm>
#include <cstdlib>
#include <map>
using namespace std;

static const bool verbose = false;

struct Magic{
    Magic(bool a, bool b, double c):combine_tensors(a), rib_stiffening(b), edge_flip_threshold(c) {}
    bool combine_tensors;
    double edge_flip_threshold;
    bool rib_stiffening;
};

static Magic magic(true, true, 0);

static Cloth::Remeshing *remeshing;
static bool plasticity;

void create_vert_sizing (const DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map);

// sizing field
struct Sizing{
    Mat2x2 M;
    Sizing() : M(Mat2x2(0)) {}
    void operator+= (const Sizing &s2) {
        M += s2.M; 
    }
    void operator-= (const Sizing &s2) {
        M -= s2.M;
    }
    void operator*= (double a) {
        M *= a;
    }
    void operator/= (double a) {
        M /= a;
    }
};

Sizing operator+ (const Sizing &s1, const Sizing &s2) {
    Sizing s = s1; s += s2; return s;
}

Sizing operator* (const Sizing &s, double a) {
    Sizing s2 = s; s2 *= a; return s2;
}
Sizing operator* (double a, const Sizing &s) {
    return s*a;
}
Sizing operator/ (const Sizing &s, double a) {
    Sizing s2 = s; s2 /= a; return s2;
}

double norm2 (const Vec2 &u, const Sizing &s) {
    return dot(u, s.M*u);
}
double norm (const Vec2 &u, const Sizing &s) {
    return sqrt(max(norm2(u,s), 0.));
}

double outer_product (const Vec2 &u, const Vec2 &v){
    // i j k
    // u0 u1 0
    // v0 v1 0
    return u[0]*v[1] - u[1]*v[0];
}

double sq(double a){
    return a*a;
}

// The algorithm

bool fix_up_mesh (DCEL &mesh, vector<Face*> &active, map<VertexId, Sizing> &vert_sizing_map, vector<HalfEdgePtr>* edges=0);

bool split_worst_edge (DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map);

bool improve_some_face (vector<Face*> &active, DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map);

void static_remesh (Cloth &cloth) {
    ::remeshing = &cloth.remeshing;
    DCEL &mesh = cloth.mesh;

    map<VertexId, Sizing> vert_sizing_map;
    for (auto &item : mesh.mVertices) {
        Sizing *sizing = new Sizing;
        sizing->M = Mat2x2(1.f/sq(remeshing->size_min));
        vert_sizing_map[item.first] = *sizing;
    }
    while (split_worst_edge(mesh, vert_sizing_map));
    vector<Face*> active = mesh.mFaces;
    while (improve_some_face(active, mesh, vert_sizing_map));
   
    vert_sizing_map.clear(); 
    //update_indices(mesh);
    cout << "now compute_ms_data" << endl;
    compute_ms_data(mesh);
    cout << "now compute_masses" << endl;
    compute_masses(mesh, 1.0);
    check_mesh_sanity(mesh);
}

void dynamic_remesh (Cloth &cloth, bool plasticity) {
    ::remeshing = &cloth.remeshing;
    ::plasticity = plasticity;
    DCEL &mesh = cloth.mesh;

    map<VertexId, Sizing> vert_sizing_map;
    create_vert_sizing(mesh, vert_sizing_map);

    vector<Face*> active = mesh.mFaces;
    fix_up_mesh(mesh, active, vert_sizing_map);

    while (split_worst_edge(mesh, vert_sizing_map));
    active = mesh.mFaces;
    while (improve_some_face(active, mesh, vert_sizing_map));

    vert_sizing_map.clear();
    //update_indices(mesh);
    cout << "now compute_ms_data" << endl;
    compute_ms_data(mesh);
    cout << "now compute_masses" << endl;
    compute_masses(mesh, 1.0);
    check_mesh_sanity(mesh);
}

// Sizing

double clamp(double input, double floor, double ceiling) {
    if (input > ceiling) input = ceiling;
    if (input < floor) input = floor;
    return input;
}

double angle (const Vec3 &n1, const Vec3 &n2) {
    return acos(clamp(dot(n1,n2),-1.,1.));
}

template <int n> Mat<n,n> sqrt (const Mat<n,n> &A) {
    Eig<n> eig = eigen_decomposition(A);
    for (int i = 0; i < n; i++)
        eig.l[i] = eig.l[i]>=0 ? sqrt(eig.l[i]) : -sqrt(-eig.l[i]);
    Mat2x2 diag;
    diag.Set(0,0, eig.l[0]);
    diag.Set(1,1, eig.l[1]);
    return eig.Q*diag*eig.Q.t();
}

template <int n> Mat<n,n> pos (const Mat<n,n> &A) {
    Eig<n> eig = eigen_decomposition(A);
    for (int i = 0; i < n; i++)
        eig.l[i] = max(eig.l[i], 0.);
    Mat2x2 diag;
    diag.Set(0,0, eig.l[0]);
    diag.Set(1,1, eig.l[1]);
    return eig.Q*diag*eig.Q.t();
}

Mat2x2 perp (const Mat2x2 &A) {
    Mat2x2 res;
    res.Set(0,0, A(1,1));
    res.Set(0,1, -A(1,0));
    res.Set(1,0, -A(0,1));
    res.Set(1,1, A(0,0));
    return res;
}

Mat2x2 curvature (const DCEL &mesh, const Face *face) {
    Mat2x2 S;
    vector<HalfEdgePtr> face_edges;
    face_edges.push_back(face->edge);
    face_edges.push_back(face->edge->nextEdge);
    face_edges.push_back(face->edge->nextEdge->nextEdge);
    for (int i = 0; i < 3; i++) {
        HalfEdgePtr e = face_edges[i];
        auto it0 = mesh.mVertices.find(e->origin),
              it1 = mesh.mVertices.find(e->nextEdge->origin);
        Vec2 e_mat = it0->second.uv - it1->second.uv;
        Vec2 t_mat = perp(normalize(e_mat));
        double theta = mesh.GetDihedralAngle(e);
        S -= 1/2. * theta * norm(e_mat) * outer(t_mat, t_mat);
    }
    S /= face->area;
    return S;
}

/* 
Mat2x2 compression_metric (const Mat2x2 &e, const Mat2x2 &S2, double c) {
    Mat2x2 D = e.t()*e - 4*sq(c)*perp(S2)*::magic.rib_stiffening;
    return pos(e*(-1) + sqrt(D))/(2*sq(c));
}
*/

/*
Mat2x2 obstacle_metric (DCEL &mesh, const Face *face, const vector<Plane> &planes) {
    Mat2x2 o = Mat2x2(0);
    vector<VertexId> face_vids;
    face_vids.push_back(face->edge->origin);
    face_vids.push_back(face->edge->nextEdge->origin);
    face_vids.push_back(face->edge->nextEdge->nextEdge->origin);
    for (int v = 0; v < 3; v++) {
        Plane p = planes[mesh.mVertices[face_vids[v]].node_ptr->id];
        if (norm2(p.second) == 0)
            continue;
        double h[3];
        for (int v1 = 0; v1 < 3; v1++){
            h[v1] = dot(mesh.mVertices[face_vids[v1]].node_ptr->x - p.first, p.second);
        }
        Vec2 dh = derivative(h[0], h[1], h[2], face);
        o += outer(dh,dh)/sq(h[v]);
    }
    return o/3.;
}
*/

Sizing compute_face_sizing (const DCEL &mesh, const Face *face) {//, const vector<Plane> &planes) {
    vector<Vertex> face_verts;
    VertexId vid0 = face->edge->origin;
    VertexId vid1 = face->edge->nextEdge->origin;
    VertexId vid2 = face->edge->nextEdge->nextEdge->origin;
    auto it0 = mesh.mVertices.find(vid0),
         it1 = mesh.mVertices.find(vid1),
         it2 = mesh.mVertices.find(vid2);
    auto node_it0 = mesh.mNodes.find(it0->second.node_id),
         node_it1 = mesh.mNodes.find(it1->second.node_id),
         node_it2 = mesh.mNodes.find(it2->second.node_id);

    Sizing s;
    //Mat2x2 Sp = curvature<PS>(mesh, face);
    Mat2x2 Sw1 = curvature(mesh, face);
    Mat3x2 Sw2 = derivative(Vec3(node_it0->second.n), 
                            Vec3(node_it1->second.n),
                            Vec3(node_it2->second.n), face);
    //Mat2x2 Mcurvp = !::plasticity ? Mat2x2(0)
    //              : (Sp.t()*Sp)/sq(remeshing->refine_angle);
    Mat2x2 Mcurvw1 = (Sw1.t()*Sw1)/sq(remeshing->refine_angle);
    Mat2x2 Mcurvw2 = (Sw2.t()*Sw2)/sq(remeshing->refine_angle);
    Mat3x2 V = derivative(Vec3(node_it0->second.vel), 
                          Vec3(node_it1->second.vel),
                          Vec3(node_it2->second.vel), face);
    Mat2x2 Mvel = (V.t()*V)/sq(remeshing->refine_velocity);
    Mat3x2 F = derivative(Vec3(node_it0->second.pos), 
                          Vec3(node_it1->second.pos),
                          Vec3(node_it2->second.pos), face);
    // Mat2x2 Mcomp = compression_metric(F.t()*F)
    //                / sq(remeshing->refine_compression);
    //Mat2x2 Mcomp = compression_metric(F.t()*F - Mat2x2(1), Sw2.t()*Sw2,
    //                                  remeshing->refine_compression);
    //Mat2x2 Mobs = (planes.empty()) ? Mat2x2(0) : obstacle_metric(face, planes);
    vector<Mat2x2> Ms(6);
    //Ms[0] = Mcurvp;
    Ms[1] = Mcurvw1;
    Ms[2] = Mcurvw2;
    Ms[3] = Mvel;
    //Ms[4] = Mcomp;
    //Ms[5] = Mobs;
    s.M = //::magic.combine_tensors ? tensor_max(Ms)
         Ms[0] + Ms[1] + Ms[2] + Ms[3] + Ms[4] + Ms[5];
    Eig<2> eig = eigen_decomposition(s.M);
    for (int i = 0; i < 2; i++)
        eig.l[i] = clamp(eig.l[i],
                         1.f/sq(remeshing->size_max),
                         1.f/sq(remeshing->size_min));
    double lmax = max(eig.l[0], eig.l[1]);
    double lmin = lmax*sq(remeshing->aspect_min);
    for (int i = 0; i < 2; i++)
        if (eig.l[i] < lmin)
            eig.l[i] = lmin;
    Mat2x2 diag;
    diag.Set(0,0, eig.l[0]);
    diag.Set(1,1, eig.l[1]);
    s.M = eig.Q*diag*eig.Q.t();
    return s;
}

static double area (const Vec2 &u0, const Vec2 &u1, const Vec2 &u2) {
    return 0.5*outer_product(u1-u0, u2-u0);
}

static double perimeter (const Vec2 &u0, const Vec2 &u1, const Vec2 &u2) {
    return norm(u0 - u1) + norm(u1 - u2) + norm(u2 - u0);
}

static double aspect (const Vec2 &u0, const Vec2 &u1, const Vec2 &u2) {
    double a = area(u0, u1, u2);
    double p = perimeter(u0, u1, u2);
    return 12*sqrt(3)*a/sq(p);
}

static double aspect (DCEL &mesh, const Face *face) {
    vector<VertexId> face_vids;
    face_vids.push_back(face->edge->origin);
    face_vids.push_back(face->edge->nextEdge->origin);
    face_vids.push_back(face->edge->nextEdge->nextEdge->origin);
    return aspect(mesh.mVertices[face_vids[0]].uv, mesh.mVertices[face_vids[1]].uv, mesh.mVertices[face_vids[2]].uv);
}

Sizing compute_vert_sizing (const DCEL &mesh, const Vertex &vert,
                            map<Face*,Sizing> &face_sizing) {
    Sizing sizing;
    vector<Face*> adj_faces = mesh.GetAdjFacesForVertex(vert.id);
    double vert_area = 0.0;
    for (Face* face : adj_faces) {
        sizing += face->area/3. * face_sizing.find((Face*)face)->second;
        vert_area += face->area/3.;
    }
    sizing /= vert_area;
    return sizing;
}

// Cache

void create_vert_sizing (const DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map) {
    map<Face*,Sizing> face_sizing;
    for (int f = 0; f < mesh.mFaces.size(); f++)
        face_sizing[mesh.mFaces[f]] = compute_face_sizing(mesh, mesh.mFaces[f]);
    for (auto &item : mesh.mVertices)
        vert_sizing_map[ item.first ] =
            compute_vert_sizing(mesh, item.second, face_sizing);
}

double edge_metric (const Vertex &vert0, const Vertex &vert1, Sizing s1, Sizing s2) {
    Vec2 du = vert0.uv - vert1.uv;
    return sqrt((norm2(du, s1) + norm2(du, s2))/2.);
}

double edge_metric (DCEL &mesh, const HalfEdgePtr edge, const map<VertexId, Sizing> &vert_sizing_map) {
    HalfEdgePtr twin = edge->twinEdge;
    const auto it0 = vert_sizing_map.find(edge->origin),
               it1 = vert_sizing_map.find(edge->nextEdge->origin);
    double m = edge_metric(mesh.mVertices[edge->origin], mesh.mVertices[edge->nextEdge->origin], it0->second, it1->second);
    if (twin) {
        const auto it2 = vert_sizing_map.find(twin->origin),
                   it3 = vert_sizing_map.find(twin->nextEdge->origin);
        m +=  edge_metric(mesh.mVertices[twin->origin], mesh.mVertices[twin->nextEdge->origin], it2->second, it3->second) ;
    }
    return twin ? m/2 : m;
}

// Helpers

template <typename T> 
void include_all (const vector<T> &u, vector<T> &v) {
    for (int i = 0; i < u.size(); i++) {
        v.push_back(u[i]);
    }
}

template <typename T>
void exclude (const T u, vector<T> &v) {
    auto it = v.begin();
    while(it != v.end()) {
        if (*it == u) {
            it = v.erase(it);
        } else {
            it++;
        }    
    }
    return;
}

template <typename T> 
void exclude_all (const vector<T> &u, vector<T> &v) {
    for (int i = 0; i < u.size(); i++) exclude(u[i], v);
}

template <typename T> 
void set_null_all (const vector<T> &u, vector<T> &v) {
    for (int i = 0; i < u.size(); i++) exclude(u[i], v);
}

void update_active (const RemeshOp &op, vector<Face*> &active) {
    exclude_all(op.removed_faces, active);
    include_all(op.added_faces, active);
}

void update_active (const vector<RemeshOp> &ops, vector<Face*> &active) {
    for (int i = 0; i < ops.size(); i++)
        update_active(ops[i], active);
}

// Fixing-upping

RemeshOp flip_edges (DCEL &mesh, vector<Face*> &active, const map<VertexId, Sizing> &vert_sizing_map);

Vertex *most_valent_vert (const vector<Face*> &faces);

// Vert *farthest_neighbor (const Vert *vert);

bool fix_up_mesh (DCEL &mesh, vector<Face*> &active, map<VertexId, Sizing> &vert_sizing_map, vector<HalfEdgePtr> *edges) {
    RemeshOp flip_ops = flip_edges(mesh, active, vert_sizing_map);
    cout << "flipped " << flip_ops.removed_edges.size() << " edges" << endl;
    update_active(flip_ops, active);
    if (edges)
        set_null_all(flip_ops.removed_edges, *edges);
    flip_ops.done();
    return !flip_ops.empty();
}

RemeshOp flip_some_edges (DCEL &mesh, vector<Face*> &active, const map<VertexId, Sizing> &vert_sizing_map);

RemeshOp flip_edges (DCEL &mesh, vector<Face*> &active, const map<VertexId, Sizing> &vert_sizing_map) {
    RemeshOp ops;
    for (int i = 0; i < 3*mesh.mVertices.size(); i++) {// don't loop without bound
        RemeshOp op = flip_some_edges(mesh, active, vert_sizing_map);
        if (op.empty())
            break;
        ops = compose(ops, op);
    }
    return ops;
}

vector<HalfEdgePtr> find_edges_to_flip (DCEL &mesh, const vector<Face*> &active, const map<VertexId, Sizing> &vert_sizing_map);
vector<HalfEdgePtr> independent_edges (DCEL &mesh, const vector<HalfEdgePtr> &edges);

bool inverted (const Face *face) {return face->area < 1e-12;}
bool degenerate (DCEL &mesh, const Face *face) {
    return aspect(mesh, face) < remeshing->aspect_min/4;}
bool any_inverted (const vector<Face*> faces) {
    for (int i=0; i<faces.size(); i++) if (inverted(faces[i])) return true;
    return false;
}
bool any_degenerate (DCEL &mesh, const vector<Face*> faces) {
    for (int i=0; i<faces.size(); i++) if (degenerate(mesh, faces[i])) return true;
    return false;
}

RemeshOp flip_some_edges (DCEL &mesh, vector<Face*> &active, const map<VertexId, Sizing> &vert_sizing_map) {
    RemeshOp ops;
    static int n_edges_prev = 0;
    vector<HalfEdgePtr> edges = independent_edges(mesh, find_edges_to_flip(mesh, active, vert_sizing_map));
    if (edges.size() == n_edges_prev) // probably infinite loop
        return ops;
    n_edges_prev = edges.size();
    for (int e = 0; e < edges.size(); e++) {
        HalfEdgePtr edge = edges[e];
        RemeshOp op = flip_edge(mesh, edge);
        op.apply(mesh);
        /*
        if (any_inverted(op.added_faces)) {
            op.inverse().apply(mesh);
            op.inverse().done();
            continue;
        }*/
        update_active(op, active);
        ops = compose(ops, op);
    }
    return ops;
}

bool should_flip (DCEL &mesh, const HalfEdgePtr edge, const map<VertexId, Sizing> &vert_sizing_map);

vector<HalfEdgePtr> find_edges_to_flip (DCEL &mesh, const vector<Face*> &active, const map<VertexId, Sizing> &vert_sizing_map){
    vector<HalfEdgePtr> edges;
    for (Face *f : active) {
        edges.push_back(f->edge);
        edges.push_back(f->edge->nextEdge);
        edges.push_back(f->edge->nextEdge->nextEdge);
    }
    vector<HalfEdgePtr> fedges;
    for (int e = 0; e < edges.size(); e++) {
        HalfEdgePtr edge = edges[e];
        if (is_seam(edge) || edge->twinEdge == nullptr || edge->label != 0)
            continue;
        if (!is_seam(edge) and edge->twinEdge and !should_flip(mesh, edge, vert_sizing_map))
            continue;
        fedges.push_back(edge);
    }
    return fedges;
}

vector<HalfEdgePtr> independent_edges (DCEL &mesh, const vector<HalfEdgePtr> &edges) {
    set<NodeId> visited_node_ids;

    vector<HalfEdgePtr> iedges;
    for (HalfEdgePtr edge : edges){
        NodeId nid0 = mesh.mVertices[edge->origin].node_id;
        NodeId nid1 = mesh.mVertices[edge->nextEdge->origin].node_id;
        if(visited_node_ids.find(nid0) == visited_node_ids.end() and
           visited_node_ids.find(nid1) == visited_node_ids.end() ){
            iedges.push_back(edge);
            visited_node_ids.insert(nid0);
            visited_node_ids.insert(nid1);
        }
    }
    return iedges;
}

double cross (const Vec2 &u, const Vec2 &v) {return u[0]*v[1] - u[1]*v[0];}

// from Bossen and Heckbert 1996
bool should_flip (DCEL &mesh, const HalfEdgePtr edge, const map<VertexId, Sizing> &vert_sizing_map) {
    VertexId vid0 = edge->origin, vid1 = edge->nextEdge->origin;
    VertexId vid2 = edge->prevEdge->origin, vid3 = edge->twinEdge->prevEdge->origin;
    const Vertex vert0 = mesh.mVertices[vid0], vert1 = mesh.mVertices[vid1],
               vert2 = mesh.mVertices[vid2], vert3 = mesh.mVertices[vid3];
    Vec2 uv0 = vert0.uv, uv1 = vert1.uv, w = vert2.uv, y = vert3.uv;
    const auto it0 = vert_sizing_map.find(vid0),
         it1 = vert_sizing_map.find(vid1),
         it2 = vert_sizing_map.find(vid2),
         it3 = vert_sizing_map.find(vid3);
    Mat2x2 M = it0->second.M;
    M += it1->second.M;
    M += it2->second.M;
    M += it3->second.M;
    M /= 4.;
    return outer_product(uv1-y, uv0-y)*dot(uv0-w, M*(uv1-w)) + dot(uv1-y, M*(uv0-y))*outer_product(uv0-w, uv1-w)
            < -::magic.edge_flip_threshold*(outer_product(uv1-y, uv0-y) + outer_product(uv0-w, uv1-w));
}

// Splitting

vector<HalfEdgePtr> find_bad_edges (DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map);

Sizing mean_vert_sizing (const Sizing &s1, const Sizing &s2);


bool split_worst_edge (DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map) {
    vector<HalfEdgePtr> edges = find_bad_edges(mesh, vert_sizing_map);
    cout << "num of edges = " << edges.size() << endl;
    int i = 0;
    for ( HalfEdgePtr edge : edges ) {
        if(find(mesh.mEdges.begin(), mesh.mEdges.end(), edge) == mesh.mEdges.end()){
            //cout << "Edge is no longer in mesh." << endl;
            continue;
        }
        cout << "edge counter = " << i++ << endl;
        VertexId vid0 = edge->origin, vid1 = edge->nextEdge->origin;
        RemeshOp op = split_edge(mesh, edge);
        op.apply(mesh);
       
        for (int v = 0; v < op.added_verts.size(); v++) {
            Vertex *vertnew = op.added_verts[v];
            vert_sizing_map[vertnew->id] = mean_vert_sizing(vert_sizing_map[vid0], vert_sizing_map[vid1]);
        }
        for (int e = 0; e < op.added_edges.size(); e++) {
            //cout << "new edge has metric " << edge_metric(mesh, op.added_edges[e], vert_sizing_map) << endl;
        }
        set_null_all(op.removed_edges, edges);
        op.done();
        if (verbose)
            cout << "Split " << vid0 << " and " << vid1 << endl;
        vector<Face*> active = op.added_faces;
        fix_up_mesh(mesh, active, vert_sizing_map, &edges);
    }
    return !edges.empty();
}

// don't use edge pointer as secondary sort key, otherwise not reproducible
struct Deterministic_sort {
    inline bool operator()(const std::pair<double,HalfEdgePtr> &left, const std::pair<double,HalfEdgePtr> &right) {
        return left.first < right.first;
    }
} deterministic_sort;

vector<HalfEdgePtr> find_bad_edges (DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map) {
    vector<pair<double,HalfEdgePtr>> edgems;
    set<HalfEdgePtr> edges_in_map;
    for ( HalfEdgePtr edge : mesh.mEdges ) {
        double m = edge_metric(mesh, edge, vert_sizing_map);
        if (m > 7 and edges_in_map.find(edge->twinEdge) == edges_in_map.end() ){
            edgems.push_back(make_pair(m, edge));
            edges_in_map.insert(edge);
        }
    }
    sort(edgems.begin(), edgems.end(), deterministic_sort);
    vector<HalfEdgePtr> edges;
    cout << "all bad edges" << endl;
    for (int i = edgems.size()-1; i >=0; i--){
        cout << "edge_metric = " << edgems[i].first << endl;
        edges.push_back(edgems[i].second);
    }
    return edges;
}

Sizing mean_vert_sizing (const Sizing &s1, const Sizing &s2) {
    Sizing sizing = s1;
    sizing += s2;
    return sizing/2.;
}

// Collapsing

vector<int> sort_edges_by_length (const Face *face);

RemeshOp try_edge_collapse (DCEL &mesh, HalfEdgePtr edge, vector<Boundary> &boundaries, map<VertexId, Sizing> &vert_sizing_map);

bool improve_some_face (vector<Face*> &active, DCEL &mesh, map<VertexId, Sizing> &vert_sizing_map) {
    cout << "improve some faces" << endl;
    vector<Boundary> boundaries = mesh.GetBoundaries();
    for (int f = 0; f < active.size(); f++) {
        cout << "for each face" << endl;
        Face *face = active[f];
        vector<HalfEdgePtr> edges;
        edges.push_back(face->edge);
        edges.push_back(face->edge->nextEdge);
        edges.push_back(face->edge->nextEdge->nextEdge);
        for (HalfEdgePtr edge : edges) {
            RemeshOp op;
            if (op.empty()) op = try_edge_collapse(mesh, edge, boundaries, vert_sizing_map);
            if (op.empty()) continue;
            op.done();
            update_active(op, active);
            vector<Face*> fix_active = op.added_faces;
            RemeshOp flip_ops = flip_edges(mesh, fix_active, vert_sizing_map);
            update_active(flip_ops, active);
            flip_ops.done();
            return true;
        }
        //remove(f--, active);
        auto it = active.begin() + f;
        active.erase(it);
        f--;
    }
    return false;
}

bool has_labeled_edges (DCEL &mesh, const Node *node);
bool can_collapse (const DCEL &mesh, const HalfEdgePtr edge, const map<VertexId, Sizing> &vert_sizing_map);
bool any_nearly_invalid (DCEL &mesh, vector<HalfEdgePtr> edges, map<VertexId, Sizing> &vert_sizing_map) {
    for (int i = 0; i < edges.size(); i++)
        if (edge_metric(mesh, edges[i], vert_sizing_map) > 0.9) return true;
    return false;
}

RemeshOp try_edge_collapse (DCEL &mesh, HalfEdgePtr edge, vector<Boundary> &boundaries, map<VertexId, Sizing> &vert_sizing_map) {
    VertexId vid0, vid1;
    vid0 = edge->origin;
    vid1 = edge->nextEdge->origin;
	
    Node &node0 = mesh.mNodes[mesh.mVertices[vid0].node_id], 
         &node1 = mesh.mNodes[mesh.mVertices[vid1].node_id];
    if ( (is_seam(edge) || edge->twinEdge==nullptr || 
          is_on_seam(mesh, boundaries, node0.vids[0]) || is_on_boundary(boundaries, node0.vids[0]) )
        || (has_labeled_edges(mesh, &node0) && !edge->label) ) {
        cout << "Try edge collapse: don't collapse because of seam or boundary." << endl;
        return RemeshOp();
    }
    if (!can_collapse(mesh, edge, vert_sizing_map)) {
        cout << "Try edge collapse: don't collapse because of sizing." << endl;
        return RemeshOp();
    }
    RemeshOp op = collapse_edge(mesh, boundaries, edge);
    op.apply(mesh);
    if (op.empty())
        return op;
    // if (any_inverted(op.added_faces) || any_degenerate(op.added_faces)
    //     || any_nearly_invalid(mesh, op.added_edges)) {
    //     op.inverse().apply(mesh);
    //     op.inverse().done();
    //     return RemeshOp();
    // }
    // delete op.removed_nodes[0]->res;
    if (verbose)
        cout << "Collapsed " << vid0 << " into " << vid1 << endl;
    return op;
}

bool has_labeled_edges (DCEL &mesh, const Node *node) {
    VertexId vid = node->vids[0];
    vector<HalfEdgePtr> adj_edges = mesh.GetAdjEdgesForVertex(vid);

    for (HalfEdgePtr edge : adj_edges) {
        if (edge->label)
            return true;
    }
    return false;
}

bool replace(VertexId vid0, VertexId vid1, vector<VertexId> &face_vids) {
    for(int i = 0; i < face_vids.size(); i++) {
        if (face_vids[i] == vid0) {
            face_vids[i] = vid1;
            return true;
        }
    }
    return false;
}

bool can_collapse (const DCEL &mesh, const HalfEdgePtr edge, const map<VertexId, Sizing> &vert_sizing_map) {
    VertexId vid0 = edge->origin, vid1 = edge->twinEdge->origin;
    vector<Face*> adj_faces = mesh.GetAdjFacesForVertex(vid0);
    for (Face* face : adj_faces) {
        vector<VertexId> face_vids;
        face_vids.push_back(face->edge->origin);
        face_vids.push_back(face->edge->nextEdge->origin);
        face_vids.push_back(face->edge->nextEdge->nextEdge->origin);
        if (find(face_vids.begin(), face_vids.end(), vid1) != face_vids.end())
            continue;
        //This line is very important!!!
        replace(vid0, vid1, face_vids);
     
        vector<Vertex> vs;
        auto it1 = mesh.mVertices.find(face_vids[0]),
             it2 = mesh.mVertices.find(face_vids[1]),
             it3 = mesh.mVertices.find(face_vids[2]);
        vs.push_back(it1->second);
        vs.push_back(it2->second);
        vs.push_back(it3->second);

        double a = outer_product(vs[1].uv - vs[0].uv, vs[2].uv - vs[0].uv)/2;
        double asp = aspect(vs[0].uv, vs[1].uv, vs[2].uv);
        if (a < 1e-6 || asp < remeshing->aspect_min)
            return false;
        
        for (int i = 0; i < 3; i++){
            const auto it0 = vert_sizing_map.find(face_vids[i]),
                       it1 = vert_sizing_map.find(face_vids[(i+1)%3]);
            if (edge_metric(vs[i], vs[(i+1)%3], it0->second, it1->second) > 0.9 * 7 )
                return false;
        }
    }
    return true;
}
