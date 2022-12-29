#ifndef RINGLIB_CPP
#define RINGLIB_CPP

#include "ringlib.h"
#include "uncached.cpp"

#define DEBUG 0

void initstate(struct RingState *s, struct RingMessage *sendbuf, struct RingMessage *recvbuf, int send_count, int recv_count, const char *name)
{
  s->next_send = 0;
  s->next_receive = 0;
  s->peer_next_receive = 0;
  s->peer_next_receive_sent = 0;
  s->sendbuf = sendbuf;
  s->recvbuf = recvbuf;
  for (int i = 0; i < NUM_MESSAGE_TYPES; i += 1) {
    s->total_sent[i] = 0;
    s->total_received[i] = 0;
  }
  s->send_count = send_count;
  s->recv_count = recv_count;
  s->name = name;
}

/* how many credits can we return? 
 * s->next_receive is the next message we haven't received yet
 * s->peer_next_receive_sent is the last next_receive we told the peer about
 * s->next_receive - s->peer_next_receive is the number of messages the peer can
 * send but we haven't told them yet.
 * The idea is that if this number gets larger than N/2 we should send an
 * otherwise empty message to update the peer
 *
 * if there is traffic in both directions, this won't trigger.  If it is needed
 * the trigger threshold should be set to N/2 or some fraction of N such that
 * the update reaches the peer before they run out of credits
 * 
 * The number of slots in the ring buffer should be enough to cover about 2 x
 * the round trip latency so that we can return credits before the peer runs out
 * when all the traffic is one-way
 */


//==============  GPU version
/* how many messages can we send? */
int gpu_ring_send_space_available(struct RingState *s)
{
  int space = ((RingN - 1) + s->peer_next_receive - s->next_send) % RingN;
  return (space);
}


int gpu_ring_internal_send_nop_p(struct RingState *s)
{
  int cr_to_send =  (RingN + s->next_receive - s->peer_next_receive_sent) % RingN;
  //  if (DEBUG) printf("send_nop_p %s: %d\n", s->name, cr_to_send);
  return (cr_to_send > NOP_THRESHOLD);
}

void gpu_ring_process_message(struct RingState *s, struct RingMessage *msg)
{
  //  if (DEBUG) printf("process %s %d (credit %d\n)\n", s->name, s->total_received, s->peer_next_receive);
  int msgtype = msg->header;
  if ((msgtype < 0) || (msgtype >= NUM_MESSAGE_TYPES)) msgtype = 0;
  s->total_received[msgtype] += 1;
  /* do what the message says */
}

/* called by ring_send when we need space to send a message */
void gpu_ring_internal_receive(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    gpu_ring_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % RingN;
  }
}


void gpu_ring_send(struct RingState *s, int type, int length, void *data)
{
  //  if (DEBUG) printf("send %s next %d\n", s->name, s->next_send);
  #if TRACE == 1
  s->wait_in_send = 17;
  #endif
  do {
    gpu_ring_internal_receive(s);
  } while (gpu_ring_send_space_available(s) == 0);
  #if TRACE == 1
  s->wait_in_send = 0;
  #endif
  ulong8 msg;
  struct RingMessage *msgp = (struct RingMessage *) &msg[0];  // local composition of message
  msgp->header = type;
  msgp->next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;
  assert (length <= MaxData);
  memcpy(&msgp->data, data, length);  // local copy
  struct RingMessage *mp = &(s->sendbuf[s->next_send]);
  ucs_ulong8((ulong8 *) mp, msg);
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
  s->next_send = (s->next_send + 1) % RingN;
  // check for array bounds !
  s->total_sent[type] += 1;
}

void gpu_send_nop(struct RingState *s)
{
  #if TRACE == 1
  s->wait_in_send_nop = 23;
  #endif
  gpu_ring_send(s, MSG_NOP, 0, NULL);
  #if TRACE == 1
  s->wait_in_send_nop = 0;
  #endif
}

/* called by users to see if any receives messages are available */
void gpu_ring_poll(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    gpu_ring_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % RingN;
    if (gpu_ring_internal_send_nop_p(s)) gpu_send_nop(s);
  }
}

void gpu_ring_drain(struct RingState *s)
{
  #if TRACE == 1
  s->wait_in_drain = 5;
  #endif
  while (s->total_received[MSG_PUT] < s->recv_count) gpu_ring_poll(s);
  #if TRACE == 1
  s->wait_in_drain = 0;
  #endif
}



//==============  CPU version
/* the number of credits available to send messages is the size of the ring
 * plus the number of messages that we know the other end has received already 
 * (peer_next_receive) minus the number we have sent
 */
int cpu_ring_send_space_available(struct RingState *s)
{
  int space = ((RingN - 1) + s->peer_next_receive - s->next_send) % RingN;
  return (space);
}

/* We should send a nop in cases that we have received a NOP_THRESHOLD worth 
 * of messages but not told the other end about it yet.  (credit return)
 */
int cpu_ring_internal_send_nop_p(struct RingState *s)
{
  int cr_to_send =  (RingN + s->next_receive - s->peer_next_receive_sent) % RingN;
  if (DEBUG) printf("send_nop_p %s: %d\n", s->name, cr_to_send);
  return (cr_to_send > NOP_THRESHOLD);
}

/* deal with an arrived message, which at the moment is only 
 * incrementing the appropriate received counter
 */

void cpu_ring_process_message(struct RingState *s, struct RingMessage *msg)
{
  //if (DEBUG) printf("process %s %d (credit %d\n)\n", s->name, s->total_received, s->peer_next_receive);
  int msgtype = msg->header;
  if ((msgtype < 0) || (msgtype >= NUM_MESSAGE_TYPES)) msgtype = 0;
  s->total_received[msgtype] += 1;
  /* do what the message says */
}

/* called by ring_send when we need space to send a message */
/* look at the next expected arriving message header and if it
 * has something in it, process the message and then clear the flag
 * and then increment the ring pointer
 */
void cpu_ring_internal_receive(struct RingState *s)
{
  if (DEBUG) printf("internal_receive %s\n", s->name);
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    cpu_ring_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % RingN;
  }
}

/* send a noop message
 * first wait for credit to send anything
 * then build a nop message
 * and then copy it to the destination ring
 * and then increment the transmit counter and transmit ring pointer
 * (for multithread, the incremnting of the ring pointer must be locked)
 */
void cpu_ring_send(struct RingState *s, int type, int length, void *data)
{
  if (DEBUG) printf("send %s next %d\n", s->name, s->next_send);
  #if TRACE == 1
  s->wait_in_send = 117;
  #endif
  do {
    cpu_ring_internal_receive(s);
  } while (cpu_ring_send_space_available(s) == 0);
  #if TRACE == 1
  s->wait_in_send = 0;
  #endif
  ulong8 msg;
  struct RingMessage *msgp = (struct RingMessage *) &msg[0];  // local composition of message
  msgp->header = type;
  msgp->next_receive = s->next_receive;
  s->peer_next_receive_sent = s->next_receive;
  assert (length <= MaxData);
  memcpy(&msgp->data, data, length);  // local copy
  struct RingMessage *mp = &(s->sendbuf[s->next_send]);
  _movdir64b(mp, &msg);
  /*
  __m512i temp = _mm512_load_epi32((void *) &msg);
  _mm512_store_si512(mp, temp);
  */
  //memcpy(mp, msgp, sizeof(struct RingMessage));

  //  _movdir64b(&(s->sendbuf[s->next_send]), &msg);   // send message (atomic!)
  s->next_send = (s->next_send + 1) % RingN;
  s->total_sent[type] += 1;
}

void cpu_send_nop(struct RingState *s)
{
  #if TRACE == 1
  s->wait_in_send_nop = 123;
  #endif
  gpu_ring_send(s, MSG_NOP, 0, NULL);
  #if TRACE == 1
  s->wait_in_send_nop = 0;
  #endif
}

/* called by users to see if any receives messages are available */
void cpu_ring_poll(struct RingState *s)
{
  /* volatile */ struct RingMessage *msg = &(s->recvbuf[s->next_receive]);
  if (DEBUG) printf("poll %s\n", s->name);
  if (msg->header != MSG_IDLE) {
    s->peer_next_receive = msg->next_receive;
    cpu_ring_process_message(s, (struct RingMessage *) msg);
    msg->header = MSG_IDLE;
    s->next_receive = (s->next_receive + 1) % RingN;
    if (cpu_ring_internal_send_nop_p(s)) cpu_send_nop(s);
  }
}

void cpu_ring_drain(struct RingState *s)
{
  #if TRACE == 1
  s->wait_in_drain = 105;
  #endif
  //  if (DEBUG) printf("drain %s received %d\n", s->name, s->total_received);
  while (s->total_received[MSG_PUT] < s->recv_count) cpu_ring_poll(s);
  #if TRACE == 1
  s->wait_in_drain = 0;
  #endif
}


#endif //! ifndef RINGLIB_CPP
