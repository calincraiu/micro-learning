#ifndef TAPE_H
#define TAPE_H

#include <vector>
#include <memory>
#include "core/Node.h"

class Tape {
public:
	Node* create_leaf(float x);
	Node* create_node(Op op, Node* left, Node* right);
	Node* create_node(Op op, Node* left, Node* right, float c);

	void backward(Node* node);
	void zero_grad();
	/// <summary>
	/// Clear all nodes from the tape.
	/// </summary>
	void clear();
	/// <summary>
	/// Clear all non-leaf nodes from the tape.
	/// </summary>
	void clear_computation_graph();

private:
	void _set_op_result(Node* node);
	std::vector<std::unique_ptr<Node>> m_nodes;
};


#endif // TAPE_H