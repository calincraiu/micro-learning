#include <cmath>

#include "core/Tape.h"
#include "core/Node.h"

Node* Tape::create_leaf(float x) {
	m_nodes.emplace_back(std::make_unique<Node>());
	Node* n = m_nodes.back().get();
    n->set_op(Op::Leaf);
	n->set_data(x);

	return n; // For forward pass
};

Node* Tape::create_node(Op op, Node* left, Node* right) {
	m_nodes.emplace_back(std::make_unique<Node>());
	Node* n = m_nodes.back().get();
	n->set_op(op);
	n->set_left(left);
	n->set_right(right);
    _set_op_result(n);

	return n; // For forward pass
}

Node* Tape::create_node(Op op, Node* left, Node* right, float c) {
    m_nodes.emplace_back(std::make_unique<Node>());
    Node* n = m_nodes.back().get();
    n->set_op(op);
    n->set_left(left);
    n->set_right(right);
    n->set_const(c);
    _set_op_result(n);

    return n; // For forward pass
}

void Tape::_set_op_result(Node* node) {

    float result = node->get_data();

    Node* left = node->get_left();
    Node* right = node->get_right();

    float l_data = 0.0f;
    float r_data = 0.0f;

    if (left) {
        l_data = left->get_data();
    }
    if (right) {
        r_data = right->get_data();
    }

    switch (node->get_op()) {

    case Op::Leaf:
        break;

    case Op::Neg:
        result = l_data * -1.0f;
        break;

    case Op::Add:
        result = l_data + r_data;
        break;

    case Op::Sub:
        result = l_data - r_data;
        break;

    case Op::Mul:
        result = l_data * r_data;
        break;

    case Op::Div:
        result = l_data / r_data;
        break;

    case Op::Pow:
        result = std::pow(l_data, r_data);
        break;

    case Op::PowConst:
        result = std::pow(l_data, node->get_const());
        break;

    case Op::Log:
        result = std::log(l_data);
        break;

    case Op::Exp:
        result = std::exp(l_data);
        break;

    case Op::ReLU:
        result = l_data > 0 ? l_data : 0;
        break;

    case Op::Tanh:
        result = std::tanh(l_data);
        break;

    default:
        break;
    }

    node->set_data(result);
}

void Tape::backward(Node* node) {
    if (!node) return;

    // Set the root gradient
    node->set_grad(node->get_grad() + 1.0f);

    // Process nodes in reverse order (reverse topological = correct for backprop)
    for (auto it = m_nodes.rbegin(); it != m_nodes.rend(); ++it) {
        Node* n = it->get();

        float grad = n->get_grad();
        // Early exit if this node has no gradient to propagate
        if (grad == 0.0f) continue;

        Node* left = n->get_left();
        Node* right = n->get_right();

        float l_data = left ? left->get_data() : 0.0f;
        float r_data = right ? right->get_data() : 0.0f;

        switch (n->get_op()) {

        case Op::Leaf:
            break;

        case Op::Neg:
            if (left) {
                left->set_grad(left->get_grad() - grad);
            }
            break;

        case Op::Add:
            if (left)  left->set_grad(left->get_grad() + grad);
            if (right) right->set_grad(right->get_grad() + grad);
            break;

        case Op::Sub:
            if (left)  left->set_grad(left->get_grad() + grad);
            if (right) right->set_grad(right->get_grad() - grad);
            break;

        case Op::Mul:
            if (left)  left->set_grad(left->get_grad() + r_data * grad);
            if (right) right->set_grad(right->get_grad() + l_data * grad);
            break;

        case Op::Div:
            if (left && r_data != 0.0f) {
                left->set_grad(left->get_grad() + grad / r_data);
            }
            if (right && r_data != 0.0f) {
                right->set_grad(right->get_grad() - grad * l_data / (r_data * r_data));
            }
            break;

        case Op::Pow: // a ^ b
        {
            if (left) {
                float deriv_a = (l_data != 0.0f || r_data == 0.0f)
                    ? r_data * std::pow(l_data, r_data - 1.0f)
                    : 0.0f;  // avoid 0^negative
                left->set_grad(left->get_grad() + deriv_a * grad);
            }
            if (right && l_data > 0.0f) {
                float deriv_b = std::pow(l_data, r_data) * std::log(l_data);
                right->set_grad(right->get_grad() + deriv_b * grad);
            }
            break;
        }

        case Op::PowConst: // a ^ c where c is constant
        {
            if (left) {
                float c = n->get_const();
                float deriv = (l_data != 0.0f)
                    ? c * (n->get_data() / l_data)
                    : 0.0f;
                left->set_grad(left->get_grad() + deriv * grad);
            }
            break;
        }

        case Op::Log:
            if (left && l_data > 0.0f) {
                left->set_grad(left->get_grad() + grad / l_data);
            }
            break;

        case Op::Exp:
            if (left) {
                left->set_grad(left->get_grad() + n->get_data() * grad);
            }
            break;

        case Op::ReLU:
            if (left) {
                float mask = (l_data > 0.0f) ? 1.0f : 0.0f;
                left->set_grad(left->get_grad() + mask * grad);
            }
            break;

        case Op::Tanh:
            if (left) {
                float t = n->get_data(); // tanh(left)
                float deriv = 1.0f - t * t; // d/dx tanh(x) = 1 - tanh^2(x)
                left->set_grad(left->get_grad() + deriv * grad);
            }
            break;

        default:
            // unknown op
            break;
        }
    }
}

void Tape::zero_grad() {
    for (auto& node_ptr : m_nodes) {
        node_ptr->set_grad(0.0f);
    }
}

void Tape::clear() {
	m_nodes.clear();
}

void Tape::clear_computation_graph() {
    auto it = m_nodes.begin();
    while (it != m_nodes.begin()) {
        if ((*it)->get_op() == Op::Leaf) {
            ++it;  // Keep leaf nodes
        }
        else {
            it = m_nodes.erase(it); // Remove computation nodes
        }
    }
}