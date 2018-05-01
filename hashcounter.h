#ifndef HASHCOUNTER_H
#define HASHCOUNTER_H

#include <map>

template<class K>
class HashCounter
{
public:
    HashCounter() { m_map = std::map<K, int>();}
    void increment(K key){
        if(m_map.find(key) == m_map.end()) {
            m_map[key] = 1;
        } else {
            m_map[key] = m_map[key] + 1;
        }
    }
    int operator [](K key){
        if(m_map.find(key) == m_map.end()) {
            return 0;
        } else {
            return m_map[key];
        }
    }
    int size() { return m_map.size(); }
    typename std::map<K,int>::iterator begin() { return m_map.begin(); }
    typename std::map<K,int>::iterator end() { return m_map.end(); }

private:
    std::map<K,int> m_map;
};

#endif // HASHCOUNTER_H
