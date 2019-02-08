/*
  Copyright ?2013 The Regents of the University of California
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

#include "remesh.h"
#include "vectors.h"
//#include "blockvectors.hpp"
//#include "geometry.hpp"
//#include "io.hpp"
//#include "magic.hpp"
//#include "util.hpp"
#include <assert.h>
#include <cstdlib>
#include <cstdio>
using namespace std;

RemeshOp RemeshOp::inverse () const {
    RemeshOp iop;
    iop.added_verts = removed_verts;
    iop.removed_verts = added_verts;
    iop.added_nodes = removed_nodes;
    iop.removed_nodes = added_nodes;
    iop.added_edges = removed_edges;
    iop.removed_edges = added_edges;
    iop.added_faces = removed_faces;
    iop.removed_faces = added_faces;
    return iop;
}

void RemeshOp::apply (DCEL &mesh) const {
    // cout << "removing " << removed_faces << ", " << removed_edges << ", " << removed_verts << " and adding " << added_verts << ", " << added_edges << ", " << added_faces << endl;
    for (int i = 0; i < removed_faces.size(); i++)
        mesh.remove(removed_faces[i]);
    for (int i = 0; i < removed_edges.size(); i++)
        mesh.remove(removed_edges[i]);
    for (int i = 0; i < removed_nodes.size(); i++)
        mesh.mNodes.erase(removed_nodes[i]);
    for (int i = 0; i < removed_verts.size(); i++)
        mesh.mVertices.erase(removed_verts[i];
    for (int i = 0; i < added_verts.size(); i++)
        mesh.mVertices[added_verts[i].id] = added_verts[i];
    for (int i = 0; i < added_nodes.size(); i++)
        mesh.mNodes[added_nodes[i].id] = added_nodes[i];
    for (int i = 0; i < added_edges.size(); i++)
        mesh.add(added_edges[i]);
    for (int i = 0; i < added_faces.size(); i++)
        mesh.add(added_faces[i]);
}

void RemeshOp::done () const {
    for (int i = 0; i < removed_verts.size(); i++)
        delete removed_verts[i];
    for (int i = 0; i < removed_nodes.size(); i++)
        delete removed_nodes[i];
    for (int i = 0; i < removed_edges.size(); i++)
        delete removed_edges[i];
    for (int i = 0; i < removed_faces.size(); i++)
        delete removed_faces[i];
}

ostream &operator<< (ostream &out, const RemeshOp &op) {
    out << "removed " << op.removed_verts << ", " << op.removed_nodes << ", "
        << op.removed_edges << ", " << op.removed_faces << ", added "
        << op.added_verts << ", " << op.added_nodes << ", " << op.added_edges
        << ", " << op.added_faces;
    return out;
}

template <typename T>
void compose_removal (T *t, vector<T*> &added, vector<T*> &removed) {
    int i = find(t, added);
    if (i != -1) {
        remove(i, added);
        delete t;
    } else {
        removed.push_back(t);
    }
}

RemeshOp compose (const RemeshOp &op1, const RemeshOp &op2) {
    RemeshOp op = op1;
    for (int i = 0; i < op2.removed_verts.size(); i++)
        compose_removal(op2.removed_verts[i], op.added_verts, op.removed_verts);
    for (int i = 0; i < op2.removed_nodes.size(); i++)
        compose_removal(op2.removed_nodes[i], op.added_nodes, op.removed_nodes);
    for (int i = 0; i < op2.removed_edges.size(); i++)
        compose_removal(op2.removed_edges[i], op.added_edges, op.removed_edges);
    for (int i = 0; i < op2.removed_faces.size(); i++)
        compose_removal(op2.removed_faces[i], op.added_faces, op.removed_faces);
    for (int i = 0; i < op2.added_verts.size(); i++)
        op.added_verts.push_back(op2.added_verts[i]);
    for (int i = 0; i < op2.added_nodes.size(); i++)
        op.added_nodes.push_back(op2.added_nodes[i]);
    for (int i = 0; i < op2.added_faces.size(); i++)
        op.added_faces.push_back(op2.added_faces[i]);
    for (int i = 0; i < op2.added_edges.size(); i++)
        op.added_edges.push_back(op2.added_edges[i]);
    return op;
}

// Fake physics for midpoint evaluation

Mat2x3 derivative_matrix (const Vec2 &u0, const Vec2 &u1, const Vec2 &u2) {
    Mat2x2 Dm = Mat2x2(u1-u0, u2-u0);
    Mat2x2 invDm = Dm.inv();
    return invDm.t()*Mat2x3::rows(Vec3(-1,1,0), Vec3(-1,0,1));
}

double area (const Vec2 &u0, const Vec2 &u1, const Vec2 &u2) {
    return wedge(u1-u0, u2-u0)/2;
}

template <int n> struct Quadratic {
    Mat<n*3,n*3> A;
    Vec<n*3> b;
    Quadratic (): A(0), b(0) {}
    Quadratic (const Mat<n*3,n*3> &A, const Vec<n*3> &b): A(A), b(b) {}
};
template <int n>
Quadratic<n> &operator*= (Quadratic<n> &q, double a) {
    q.A *= a; q.b *= a; return q;}
template <int n>
Quadratic<n> &operator+= (Quadratic<n> &q, const Quadratic<n> &r) {
    q.A += r.A; q.b += r.b; return q;}
template <int n>
ostream &operator<< (ostream &out, const Quadratic<n> &q) {out << "<" << q.A << ", " << q.b << ">"; return out;}

template <int m, int n, int p, int q>
Mat<m*p,n*q> kronecker (const Mat<m,n> &A, const Mat<p,q> &B) {
    Mat<m*p,n*q> C;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                for (int l = 0; l < q; l++)
                    C(i*p+k,j*q+l) = A(i,j)*B(k,l);
    return C;
}

template <int m> Mat<m,1> colmat (const Vec<m> &v) {
    Mat<1,m> A; for (int i = 0; i < m; i++) A(i,0) = v[i]; return A;}
template <int n> Mat<1,n> rowmat (const Vec<n> &v) {
    Mat<1,n> A; for (int i = 0; i < n; i++) A(0,i) = v[i]; return A;}

Quadratic<3> stretching (const Vertex *vert0, const Vertex *vert1,
                         const Vertex *vert2) {
    const Vec2 &uv0 = vert0->uv, &uv1 = vert1->uv, &uv2 = vert2->uv;
    const Vec3 &x0 = vert0->node_ptr->pos, &x1 = vert1->node_ptr->pos,
               &x2 = vert2->node_ptr->pos;
    Mat2x3 D = derivative_matrix(uv0, uv1, uv2);
    Mat3x2 F = Mat3x3(x0,x1,x2)*D.t(); // = (D * Mat3x3(x0,x1,x2).t()).t()
    Mat2x2 G = (F.t()*F - Mat2x2(1))/2.;
    // eps = 1/2(F'F - I) = 1/2([x_u^2 & x_u x_v \\ x_u x_v & x_v^2] - I)
    // e = 1/2 k0 eps00^2 + k1 eps00 eps11 + 1/2 k2 eps11^2 + 1/2 k3 eps01^2
    // grad e = k0 eps00 grad eps00 + ...
    //        = k0 eps00 Du' x_u + ...
    Vec3 du = D.row(0), dv = D.row(1);
    Mat<3,9> Du = kronecker(rowmat(du), Mat3x3(1)),
             Dv = kronecker(rowmat(dv), Mat3x3(1));
    const Vec3 &xu = F.col(0), &xv = F.col(1); // should equal Du*mat_to_vec(X)
    Vec<9> fuu = Du.t()*xu, fvv = Dv.t()*xv, fuv = (Du.t()*xv + Dv.t()*xu)/2.;
    Vec<4> k;
    k[0] = 1;
    k[1] = 0;
    k[2] = 1;
    k[3] = 1;
    Vec<9> grad_e = k[0]*G(0,0)*fuu + k[2]*G(1,1)*fvv
                  + k[1]*(G(0,0)*fvv + G(1,1)*fuu) + k[3]*G(0,1)*fuv;
    Mat<9,9> hess_e = k[0]*(outer(fuu,fuu) + max(G(0,0),0.)*Du.t()*Du)
                    + k[2]*(outer(fvv,fvv) + max(G(1,1),0.)*Dv.t()*Dv)
                    + k[1]*(outer(fuu,fvv) + max(G(0,0),0.)*Dv.t()*Dv
                            + outer(fvv,fuu) + max(G(1,1),0.)*Du.t()*Du)
                    + k[3]*(outer(fuv,fuv));
    // ignoring k[3]*G(0,1)*(Du.t()*Dv+Dv.t()*Du)/2.) term
    // because may not be positive definite
    double a = area(u0, u1, u2);
    return Quadratic<3>(a*hess_e, a*grad_e);
}

double area (const Vec3 &x0, const Vec3 &x1, const Vec3 &x2) {
    return norm(cross(x1-x0, x2-x0))/2;
}
Vec3 normal (const Vec3 &x0, const Vec3 &x1, const Vec3 &x2) {
    return normalize(cross(x1-x0, x2-x0));
}
double dihedral_angle (const Vec3 &e, const Vec3 &n0, const Vec3 &n1) {
    double cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
    return -atan2(sine, cosine);
}

template <Space s>
Quadratic<4> bending (double theta0, const Vertex *vert0, const Vertex *vert1,
                                     const Vertex *vert2, const Vertex *vert3) {
    const Vec3 &x0 = (vert0->node_ptr->pos), &x1 = (vert1->node_ptr->pos),
               &x2 = (vert2->node_ptr->pos), &x3 = (vert3->node_ptr->pos);
    Vec3 n0 = normal(x0,x1,x2), n1 = normal(x1,x0,x3);
    double theta = dihedral_angle(normalize(x1-x0), n0, n1);
    double l = norm(x0-x1);
    double a0 = area(x0,x1,x2), a1 = area(x1,x0,x3);
    double h0 = 2*a0/l, h1 = 2*a1/l;
    double w_f0v0 = dot(x2-x1, x0-x1)/sq(l),
           w_f0v1 = 1 - w_f0v0,
           w_f1v0 = dot(x3-x1, x0-x1)/sq(l),
           w_f1v1 = 1 - w_f1v0;
    Vec<12> dtheta = mat_to_vec(Mat<3,4>(-(w_f0v0*n0/h0 + w_f1v0*n1/h1),
                                         -(w_f0v1*n0/h0 + w_f1v1*n1/h1),
                                         n0/h0,
                                         n1/h1));
    double ke = 1;
    double shape = 1;//sq(l)/(2*(a0 + a1));
    return Quadratic<4>((a0+a1)/4*ke*shape*outer(dtheta, dtheta)/2.,
                        (a0+a1)/4*ke*shape*(theta-theta0)*dtheta/2.);
}

template <int n> Quadratic<1> restrict (const Quadratic<n> &q, int k) {
    Quadratic<1> r;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            r.A(i,j) = q.A(k*3+i, k*3+j);
        r.b[i] = q.b[k*3+i];
    }
    return r;
}

void set_midpoint_position (DCEL &mesh, const HalfEdgePtr edge, Vertex *vnew[2], Node *node) {
    Quadratic<1> qs, qb;
    for (int i = 0; i < 2; i++) {
        if (!edge->adjf[i])
            continue;
        VertexId vid0 = edge->origin, vid1 = edge->nextEdge->origin;
        const Vertex *v0 = edge_vert(edge, i, i),
                   *v1 = edge_vert(edge, i, 1-i),
                   *v2 = edge_opp_vert(edge, i),
                   *v = vnew[i];
        qs += restrict(stretching(v0, v, v2), 1);
        qs += restrict(stretching(v, v1, v2), 0);
        qb += restrict(bending(0, v, v2, v0, v1), 0);
        // if (S == WS) REPORT(qb);
        const HalfEdgePtr e;
        e = mesh.FindHalfEdgeByVertices(v0->id, v2->id);
        if (const Vertex *v4 = edge_opp_vert(e, e->n[0]==v0->node ? 0 : 1))
            qb += restrict(bending(e->theta_ideal, v0, v2, v4, v), 3);
        // if (S == WS) REPORT(qb);
        e = mesh.FindHalfEdgeByVertices(v1->id, v2->id);
        if (const Vertex *v4 = edge_opp_vert(e, e->n[0]==v1->node ? 1 : 0))
            qb += restrict(bending(e->theta_ideal, v1, v2, v, v4), 2);
        // if (S == WS) REPORT(qb);
    }
    if (edge->face && edge->twinEdge->face) {
        const Vertex *v2 = edge_opp_vert(edge, 0), *v3 = edge_opp_vert(edge, 1);
        double theta = edge->theta_ideal;
        qb += restrict(bending(theta, edge_vert(edge, 0, 0), vnew[0],
                                         v2, v3), 1);
        // if (S == WS) REPORT(qb);
        qb += restrict(bending(theta, vnew[1], edge_vert(edge, 1, 1),
                                         v2, v3), 0);
        // if (S == WS) REPORT(qb);
    }
    Quadratic<1> q;
    q += qs;
    q += qb;
    q.A += Mat3x3(1e-3);
    // if (S == WS) {
    //     REPORT(pos<S>(node));
    //     REPORT(qs.A);
    //     REPORT(qs.b);
    //     REPORT(qb.A);
    //     REPORT(qb.b);
    //     REPORT(-q.A.inv()*q.b);
    // }
    pos(node) -= q.A.inv()*q.b;
}

// The actual operations

int combine_label (int l0, int l1) {return (l0==l1) ? l0 : 0;}

//Originally edge is associated with 2 nodes.
//However, in my implementation, edge is associated with 2 vertices.
RemeshOp split_edge (DCEL &mesh, HalfEdgePtr edge) {
    RemeshOp op;
    VertexId vid0 = edge->origin, vid1 = edge->nextEdge->origin;
    Vertex v0 = mesh.mVertices[vid0], v1 = mesh.mVertices[vid1];
    VertexId new_vid = 0;
    FaceId fid = 0;
    Vertex *new_v = new Vertex(new_vid, (v0.uv + v1.uv)/2);
    Node *node0 = mesh.mVertices[vid0].node_ptr;
         *node1 = mesh.mVertices[vid1].node_ptr;
         *new_node = new Node((node0->y + node1->y)/2.,
                          (node0->pos + node1->pos)/2.,
                          (node0->vel + node1->vel)/2.,
                          combine_label(node0->label, node1->label));
    node->acceleration = (node0->acceleration + node1->acceleration)/2.;
    op.added_nodes.push_back(node);
    op.removed_edges.push_back(edge);
    
    Vertex *vnew[2] = {NULL, NULL};
    HalfEdgePtr new_e0 = new HalfEdge(vid0, edge->theta_ideal);
    new_e0->setPrev(edge->prevEdge); 
    
    HalfEdgePtr new_e1 = new HalfEdge(new_v->id, edge->theta_ideal);
    new_e1->setNext(edge->nextEdge);
        
    op.added_edges.push_back(new_e0);
    op.added_edges.push_back(new_e1);

    {//Add and connect 2 new faces on one side
	    Face *f = edge->face;
	    op.removed_faces.push_back(f);
	    Face *new_f0 = new Face(fid++);
	    Face *new_f1 = new Face(fid++);
	    new_e0->face = new_f0;
	    new_e1->face = new_f1;
	    new_f0->edge = new_e0;
		new_f1->edge = new_e1;
	    mesh.mFaces.push_back(new_f0);
	    mesh.mFaces.push_back(new_f1);
       
		HalfEdgePtr new_bridge_edge = new HalfEdge(new_vid);
		HalfEdgePtr new_bridge_twin = new HalfEdge(edge->prevEdge->origin);
		new_bridge_edge->setTwin(new_bridge_twin);
				
		new_bridge_edge->face = new_f0;
		new_bridge_edge->setPrev(new_e0);
		new_bridge_edge->setNext(edge->prevEdge);
		
		new_bridge_twin->face = new_f1;
		new_bridge_twin->setPrev(edge->nextEdge);
		new_bridge_twin->setNext(new_e1);
		
        if (s == 0 || is_seam_or_boundary(edge)) {            
            connect(vnew[s], node);
            op.added_verts.push_back(vnew[s]);
        } else {
        	vnew[s] = vnew[0];
		}                   
        op.added_edges.push_back(new_bridge_edge);
        op.added_edges.push_back(new_bridge_twin);
        
        op.added_faces.push_back(new_f0);
        op.added_faces.push_back(new_f1);
    }
    
    if (edge->twinEdge != NULL) {//Add and connect 2 faces on the other side
    	HalfEdgePtr new_twin0 = new HalfEdge(new_v->id, edge->theta_ideal);
        new_e0->setTwin(new_twin0);
        HalfEdgePtr new_twin1 = new HalfEdge(vid1, edge->theta_ideal);
    	new_e1->setTwin(new_twin1);
    	
    	op.added_edges.push_back(new_twin0);
    	op.added_edges.push_back(new_twin1);
        
    	HalfEdgePtr twin = edge->twinEdge;
        Face *f = twin->face;
	    op.removed_faces.push_back(f);
	    Face *new_f2 = new Face(fid++);
	    Face *new_f3 = new Face(fid++);
	    new_twin0->face = new_f2;
	    new_twin1->face = new_f3;
	    new_f2->edge = new_twin0;
            new_f3->edge = new_twin1;
       
        HalfEdgePtr new_bridge_edge2 = new HalfEdge(new_vid);
	HalfEdgePtr new_bridge_twin2 = new HalfEdge(twin->prevEdge->origin);
	new_bridge_edge2->setTwin(new_bridge_twin2);
				
	new_bridge_edge2->face = new_f2;
	new_bridge_edge2->setPrev(twin->nextEdge);
	new_bridge_edge2->setNext(new_twin0);
		
	new_bridge_twin2->face = new_f3;
	new_bridge_twin2->setPrev(new_twin1);
	new_bridge_twin2->setNext(twin->prevEdge);
		
        if (s == 0 || is_seam_or_boundary(edge)) {            
            connect(vnew[s], node);
            op.added_verts.push_back(vnew[s]);
        } else {
            vnew[s] = vnew[0];
	}  
		                 
        op.added_edges.push_back(new_bridge_edge2);
        op.added_edges.push_back(new_bridge_twin2);        
        op.added_faces.push_back(new_f2);
        op.added_faces.push_back(new_f3);                    
    }
    if (!::magic.preserve_creases) {
        set_midpoint_position<PS>(edge, vnew, node);
        set_midpoint_position<WS>(edge, vnew, node);
    }
    return op;
}

//the orientation of edge gives the orientation of collapse
RemeshOp collapse_edge (DCEL& mesh, HalfEdgePtr edge) {
    RemeshOp op;
    VertexId vid0 = edge->origin, vid1 = edge->nextEdge->origin;
    Node *node0 = mesh.mVertices[vid0].node_ptr, *node1 = mesh.mVertices[vid1].node_ptr;
    
    op.removed_nodes.push_back(node0);
    op.removed_verts.push_back(vid0);
    vector<HalfEdgePtr> adj_edges = mesh.GetAdjEdgesForVertex(node0->id);
    for (int e = 0; e < adj_edges.size(); e++) {
        HalfEdgePtr edge = adj_edges[e];
        op.removed_edges.push_back(edge);
        op.removed_edges.push_back(edge->twinEdge);
        VertexId vid2 = edge->nextEdge->origin;
        if (vid2 != vid1 && !mesh.FindEdgeByVertices(vid1, vid2)){
            HalfEdgePtr new_e = new HalfEdge(vid1, edge->theta_ideal);
            HalfEdgePtr new_twin = new HalfEdge(vid2, edge->theta_ideal);
            new_e->setTwin(new_twin);
            op.added_edges.push_back(new_e);
            op.added_edges.push_back(new_twin);
        }
    }
    for (int e = 0; e < adj_edges.size(); e++) {
        HalfEdgePtr edge = adj_edges[e];
        Face *face = edge->face;
        op.removed_faces.push_back(face);
        vector<VertexId> face_vids;
        face_vids.push_back(edge->origin);
        face_vids.push_back(edge->nextEdge->origin);
        face_vids.push_back(edge->nextEdge->nextEdge->origin);
        if (face_vids.find(vid1) == face_vids.end()) {
            VertexId vids[3]; 
	        vids[0] = vid1;
            int i = 0;
            while( face_vids[i] != vid0 ){
                i = (i+1)%3;
            } 
            i = (i+1)%3;
	        vids[1] = face_vids[i];
            i = (i+1)%3;
	        vids[2] = face_vids[i];

            Face* new_f = new Face(fid++);
            HalfEdgePtr new_e0 = new HalfEdge(vids[0]);
            HalfEdgePtr new_e1 = new HalfEdge(vids[1]);
            HalfEdgePtr new_e2 = new HalfEdge(vids[2]);
            new_e0->face = new_f;
            new_e1->face = new_f;
            new_e2->face = new_f; 
            new_f->edge = new_e0;
            new_f->label = face->label;
            op.added_faces.push_back(new_f);
        }
    }
    return op;
}

RemeshOp flip_edge (Edge* edge) {
    RemeshOp op;
    Vert *vert0 = edge_vert(edge, 0, 0), *vert1 = edge_vert(edge, 1, 1),
         *vert2 = edge_opp_vert(edge, 0), *vert3 = edge_opp_vert(edge, 1);
    Face *face0 = edge->adjf[0], *face1 = edge->adjf[1];
    op.removed_edges.push_back(edge);
    op.added_edges.push_back(new HalfEdge(vert2->node, vert3->node,
                                      -edge->theta_ideal, edge->label));
    op.removed_faces.push_back(face0);
    op.removed_faces.push_back(face1);
    op.added_faces.push_back(new Face(vert0, vert3, vert2, face0->label));
    op.added_faces.push_back(new Face(vert1, vert2, vert3, face1->label));
    return op;
}

